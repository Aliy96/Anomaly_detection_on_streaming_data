#!/usr/bin/env python3
"""
Real-time anomaly detector using AnomalyTransformer models.
FIXED: Properly preserves message metadata (spacecraft, anomaly_class) through buffering.
"""
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from kafka import KafkaConsumer
from collections import defaultdict

import sys
from datetime import datetime

# Display config
QUIET_MODE = False  # Set True for log-friendly output (no colors)
ALERT_COLORS = {
    "critical": "\033[1;31m",   # Bright red
    "warning": "\033[1;33m",    # Bright yellow
    "normal": "\033[0m",
    "reset": "\033[0m"
}

def format_anomaly_class(raw_class):
    """Normalize variable-length class strings"""
    if not raw_class or raw_class == "unknown":
        return "normal"
    # Extract first class name from list representation
    clean = str(raw_class).strip("[]'\" ").split(",")[0].strip()
    return clean[:10]  # Truncate to 10 chars max

def get_alert_symbol(score, threshold):
    if score > threshold * 1.5:
        return ("🚨", "critical") if not QUIET_MODE else ("CRITICAL", "critical")
    elif score > threshold:
        return ("⚠️", "warning") if not QUIET_MODE else ("WARNING", "warning")
    return ("✅", "normal") if not QUIET_MODE else ("NORMAL", "normal")
    
# Config
KAFKA_BROKER = "localhost:9092"
TOPIC = "telemetry-stream"
GROUP_ID = "transformer-detector-v1"
MODEL_DIR = Path("models/anomaly_transformer")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

print(f"🔍 AnomalyTransformer Detector | Broker: {KAFKA_BROKER}")
print(f"   Models: {MODEL_DIR} | Device: {DEVICE}\n")

# -----------------------------
# EXACT AnomalyTransformer Architecture (MUST match training)
# -----------------------------
class AnomalyAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.u = nn.Parameter(torch.randn(n_heads, 1, 1) * 0.1)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        Q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn_series = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_series = F.softmax(attn_series, dim=-1)
        
        positions = torch.arange(L, device=x.device).float()
        prior = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        prior = prior.unsqueeze(0).unsqueeze(0).expand(B, self.n_heads, L, L)
        attn_prior = torch.exp(-prior ** 2 / (2 * (self.u ** 2 + 1e-6)))
        attn_prior = attn_prior / (attn_prior.sum(dim=-1, keepdim=True) + 1e-6)
        
        discrepancy = torch.abs(attn_series - attn_prior).mean(dim=(1, 2))
        out = torch.matmul(attn_series, V).transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        return out, discrepancy


class AnomalyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = AnomalyAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor):
        attn_out, discrepancy = self.attention(x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x, discrepancy


class AnomalyTransformerForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 8,
        window_size: int = 100,
        n_pred: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.window_size = window_size
        self.n_pred = n_pred
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            AnomalyTransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.forecast_head = nn.Linear(d_model, n_pred)
        
    def forward(self, x: torch.Tensor):
        x_proj = self.input_proj(x)
        total_discrepancy = 0
        for layer in self.layers:
            x_proj, disc = layer(x_proj)
            total_discrepancy += disc
        avg_discrepancy = total_discrepancy / len(self.layers)
        context = x_proj.mean(dim=1)
        forecast = self.forecast_head(context)
        return forecast, avg_discrepancy


# -----------------------------
# ENHANCED Model Cache (deduplicated loading messages)
# -----------------------------
class ModelCache:
    def __init__(self, model_dir, max_models=8):
        self.model_dir = model_dir
        self.max_models = max_models
        self.cache = {}
        self.access = []
        self.loaded_channels = set()  # Track already-loaded channels
    
    def get(self, chan_id):
        if chan_id in self.cache:
            self.access.remove(chan_id)
            self.access.append(chan_id)
            return self.cache[chan_id]
        
        # Only show load message once per channel
        if chan_id not in self.loaded_channels:
            print(f"📦 Loading model for {chan_id}...", file=sys.stderr, flush=True)
            self.loaded_channels.add(chan_id)
        
        path = self.model_dir / f"{chan_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Model missing: {path}")
        
        ckpt = torch.load(path, map_location=DEVICE)
        model = AnomalyTransformerForecaster(
            input_dim=ckpt["input_dim"],
            d_model=128,
            n_layers=3,
            n_heads=8,
            window_size=ckpt["l_s"],
            n_pred=ckpt["n_pred"],
            dropout=0.2
        ).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        threshold = ckpt["threshold"]
        
        # LRU management
        if len(self.cache) >= self.max_models:
            lru = self.access.pop(0)
            del self.cache[lru]
        self.cache[chan_id] = (model, threshold)
        self.access.append(chan_id)
        
        if chan_id in self.loaded_channels:
            print(f"✅ {chan_id} ready (threshold={threshold:.4f})", file=sys.stderr, flush=True)
        return model, threshold

# -----------------------------
# Detection Engine (FIXED: preserves metadata)
# -----------------------------
class Detector:
    def __init__(self, model_cache):
        self.cache = model_cache
        # Buffer stores FULL messages (not just windows) to preserve metadata
        self.buffer = defaultdict(list)  # chan_id -> [messages]
        self.stats = defaultdict(lambda: {"windows": 0, "alerts": 0})
    
    @torch.no_grad()
    def process(self, msg):
        chan_id = msg["chan_id"]
        self.buffer[chan_id].append(msg)  # Store entire message with metadata
        
        alerts = []
        if len(self.buffer[chan_id]) >= BATCH_SIZE:
            alerts = self._detect_batch(chan_id)
        
        return alerts
    
    def _detect_batch(self, chan_id):
        batch_msgs = self.buffer[chan_id]
        self.buffer[chan_id] = []
        
        timestamps = [m["timestamp"] for m in batch_msgs]
        windows = [m["window"] for m in batch_msgs]
        spacecrafts = [m.get("spacecraft", "unknown") for m in batch_msgs]
        anomaly_classes = [m.get("anomaly_class", "unknown") for m in batch_msgs]
        
        try:
            model, threshold = self.cache.get(chan_id)
            X = torch.tensor(windows, dtype=torch.float32).to(DEVICE)
            
            start = time.time()
            pred, discrepancy = model(X)
            latency = (time.time() - start) * 1000
            
            # Combined anomaly score
            recon_error = ((X[:, -1, 0] - pred.squeeze(1)) ** 2).cpu().numpy()
            disc_score = discrepancy.mean(dim=1).cpu().numpy()
            combined_score = np.sqrt(recon_error * disc_score + 1e-8)
            
            flags = combined_score > threshold
            alerts = []
            for i, is_anomaly in enumerate(flags):
                if is_anomaly:
                    alerts.append({
                        "chan_id": chan_id,
                        "timestamp": timestamps[i],
                        "score": float(combined_score[i]),
                        "recon_error": float(recon_error[i]),
                        "discrepancy": float(disc_score[i]),
                        "threshold": threshold,
                        "latency_ms": latency / len(batch_msgs),
                        "spacecraft": spacecrafts[i],
                        "anomaly_class": anomaly_classes[i]
                    })
                    self.stats[chan_id]["alerts"] += 1
            
            self.stats[chan_id]["windows"] += len(batch_msgs)
            return alerts
            
        except Exception as e:
            print(f"⚠️  Error processing {chan_id}: {e}")
            import traceback
            traceback.print_exc()
            return []


# -----------------------------
# Main Consumer Loop
# -----------------------------
def main():
    cache = ModelCache(MODEL_DIR)
    detector = Detector(cache)
    
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        group_id=GROUP_ID,
        value_deserializer=lambda m: json.loads(m.decode()),
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        consumer_timeout_ms=30000
    )
    print("✅ Connected to Kafka\n")
    
    print("📡 Listening for telemetry streams...\n")
    # AFTER: print("✅ Connected to Kafka\n")
# REPLACE the alert printing section with:

    print(f"{'='*100}")
    print(f"{'TIME':<9} {'CHANNEL':<8} {'TIMESTAMP':<9} {'SCORE':<8} {'RECON':<8} {'DISC':<8} {'THR':<8} {'LAT':<7} {'SPACECRAFT':<10} {'CLASS':<10} {'ALERT'}")
    print(f"{'='*100}")

    # Dashboard state
    last_dashboard = time.time()
    DASHBOARD_INTERVAL = 5.0  # seconds

    try:
        msg_count = 0
        for message in consumer:
            msg_count += 1
            msg = message.value
            alerts = detector.process(msg)
            
            # Display alerts with clean formatting
            for alert in alerts:
                ts_str = datetime.now().strftime('%H:%M:%S')
                symbol, severity = get_alert_symbol(alert["score"], alert["threshold"])
                anomaly_class = format_anomaly_class(msg.get("anomaly_class", "unknown"))
                
                # Build colorized output
                color = ALERT_COLORS[severity] if not QUIET_MODE else ""
                reset = ALERT_COLORS["reset"] if not QUIET_MODE else ""
                
                line = (f"{color}{ts_str:<9} {alert['chan_id']:<8} {alert['timestamp']:<9} "
                    f"{alert['score']:>7.4f} {alert['recon_error']:>7.4f} {alert['discrepancy']:>6.4f} "
                    f"{alert['threshold']:>6.4f} {alert['latency_ms']:>6.1f}ms "
                    f"{msg.get('spacecraft', 'unknown'):<10} {anomaly_class:<10} {symbol}{reset}")
                print(line, flush=True)
            
            # Periodic dashboard (non-intrusive)
            now = time.time()
            if now - last_dashboard >= DASHBOARD_INTERVAL and detector.stats:
                last_dashboard = now
                total_windows = sum(s["windows"] for s in detector.stats.values())
                total_alerts = sum(s["alerts"] for s in detector.stats.values())
                alert_rate = total_alerts / max(total_windows, 1) * 100
                
                print(f"\n{ALERT_COLORS['warning'] if not QUIET_MODE else ''}📊 LIVE DASHBOARD "
                    f"(updated {datetime.now().strftime('%H:%M:%S')}){ALERT_COLORS['reset'] if not QUIET_MODE else ''}")
                print(f"   Total windows processed: {total_windows:>7}")
                print(f"   Total alerts triggered : {total_alerts:>7} ({alert_rate:>5.2f}%)")
                print(f"   Active channels        : {len(detector.stats):>7}")
                print(f"{'─'*100}\n", flush=True)
                
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping consumer...", flush=True)
    finally:
        consumer.close()
        # Final summary
        print(f"\n{'='*100}")
        print("✅ STREAMING COMPLETE - FINAL STATISTICS")
        print(f"{'='*100}")
        total_windows = sum(s["windows"] for s in detector.stats.values())
        total_alerts = sum(s["alerts"] for s in detector.stats.values())
        print(f"{'Channel':<10} {'Windows':<10} {'Alerts':<10} {'Alert Rate':<12} {'Threshold'}")
        print(f"{'-'*100}")
        for cid, s in sorted(detector.stats.items()):
            rate = s['alerts'] / max(s['windows'], 1) * 100
            model, thresh = cache.get(cid)  # Get threshold for display
            print(f"{cid:<10} {s['windows']:<10} {s['alerts']:<10} {rate:>6.2f}%      {thresh:>8.4f}")
        print(f"{'-'*100}")
        print(f"{'TOTAL':<10} {total_windows:<10} {total_alerts:<10} {total_alerts/max(total_windows,1)*100:>6.2f}%")
        print(f"{'='*100}\n")
        
if __name__ == "__main__":
    main()