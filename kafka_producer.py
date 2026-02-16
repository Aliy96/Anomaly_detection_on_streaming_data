#!/usr/bin/env python3
"""
Stream NASA test data windows to Kafka for real-time anomaly detection.
Compatible with AnomalyTransformer models (models/anomaly_transformer/).
"""
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from kafka import KafkaProducer
import threading
from queue import Queue
# Config
TEST_DIR = Path("DataSet/data/data/test")
LABELS_PATH = Path("DataSet/labeled_anomalies.csv")
KAFKA_BROKER = "localhost:9092"
TOPIC = "telemetry-stream"
WINDOW_SIZE = 100      # Must match training L_S
STREAM_DELAY = 0.05   # Seconds per window (adjust for speed)
MAX_CHANNELS = 12       # Set to None for all channels

def load_labels():
    df = pd.read_csv(LABELS_PATH)
    return {row["chan_id"]: {"spacecraft": row["spacecraft"], "class": row["class"]} 
            for _, row in df.iterrows()}

def stream_channel(chan_id, producer, metadata, total, idx):
    test_path = TEST_DIR / f"{chan_id}.npy"
    if not test_path.exists():
        print(f"[{idx}/{total}] ✗ {chan_id}: file not found")
        return 0
    
    try:
        data = np.load(test_path).astype(np.float32)
        if data.ndim != 2 or data.shape[1] < 2:
            print(f"[{idx}/{total}] ✗ {chan_id}: invalid shape {data.shape}")
            return 0
        
        # Forecasting setup: predict col 0 using cols 1..F-1
        windows = []
        for i in range(len(data) - WINDOW_SIZE):
            window = data[i:i+WINDOW_SIZE, 1:].tolist()  # [L_S, F-1]
            windows.append({
                "chan_id": chan_id,
                "timestamp": i + WINDOW_SIZE - 1,
                "window": window,
                "spacecraft": metadata.get(chan_id, {}).get("spacecraft", "unknown"),
                "anomaly_class": metadata.get(chan_id, {}).get("class", "unknown")
            })
        
        # Stream windows
        sent = 0
        for msg in windows:
            producer.send(TOPIC, key=chan_id.encode(), value=json.dumps(msg).encode())
            sent += 1
            time.sleep(STREAM_DELAY)
            
            if sent % 500 == 0:
                print(f"[{idx}/{total}] {chan_id}: {sent}/{len(windows)} windows", end='\r')
        
        print(f"[{idx}/{total}] ✓ {chan_id}: {sent} windows streamed")
        return sent
        
    except Exception as e:
        print(f"[{idx}/{total}] ✗ {chan_id} error: {e}")
        return 0


def stream_channel_thread(chan_id, producer, metadata, total, idx, progress_queue):
    """Thread-safe channel streaming"""
    test_path = TEST_DIR / f"{chan_id}.npy"
    if not test_path.exists():
        progress_queue.put((idx, chan_id, "error", "file not found"))
        return 0
    
    try:
        data = np.load(test_path).astype(np.float32)
        if data.ndim != 2 or data.shape[1] < 2:
            progress_queue.put((idx, chan_id, "error", f"invalid shape {data.shape}"))
            return 0
        
        windows = []
        for i in range(len(data) - WINDOW_SIZE):
            window = data[i:i+WINDOW_SIZE, 1:].tolist()
            windows.append({
                "chan_id": chan_id,
                "timestamp": i + WINDOW_SIZE - 1,
                "window": window,
                "spacecraft": metadata.get(chan_id, {}).get("spacecraft", "unknown"),
                "anomaly_class": metadata.get(chan_id, {}).get("class", "unknown")
            })
        
        sent = 0
        local_producer = KafkaProducer(  # Each thread gets its own producer
            bootstrap_servers=[KAFKA_BROKER],
            acks='all',
            retries=3,
            request_timeout_ms=5000
        )
        
        for msg in windows:
            local_producer.send(TOPIC, key=chan_id.encode(), value=json.dumps(msg).encode())
            sent += 1
            time.sleep(STREAM_DELAY)
            if sent % 500 == 0:
                progress_queue.put((idx, chan_id, "progress", sent, len(windows)))
        
        local_producer.flush()
        local_producer.close()
        progress_queue.put((idx, chan_id, "complete", sent))
        return sent
        
    except Exception as e:
        progress_queue.put((idx, chan_id, "error", str(e)))
        return 0

def progress_monitor(progress_queue, total_channels):
    """Display progress from all threads"""
    completed = 0
    results = {}
    while completed < total_channels:
        item = progress_queue.get()
        idx, chan_id, status, *rest = item
        
        if status == "progress":
            sent, total = rest
            print(f"[{idx}/{total_channels}] {chan_id}: {sent}/{total} windows", end='\r')
        elif status == "complete":
            sent = rest[0]
            completed += 1
            results[chan_id] = sent
            print(f"[{idx}/{total_channels}] ✓ {chan_id}: {sent} windows streamed")
        elif status == "error":
            msg = rest[0]
            completed += 1
            print(f"[{idx}/{total_channels}] ✗ {chan_id} error: {msg}")
    
    return sum(results.values())

def main():
    print(f"🚀 Kafka Producer | Broker: {KAFKA_BROKER} | Topic: {TOPIC}")
    print(f"   Window size: {WINDOW_SIZE} | Delay: {STREAM_DELAY*1000:.1f}ms/window")
    print(f"   Parallel channels: {MAX_CHANNELS or 'ALL'}\n")
    
    metadata = load_labels()
    chan_ids = list(metadata.keys())
    if MAX_CHANNELS:
        chan_ids = chan_ids[:MAX_CHANNELS]
    
    print(f"📡 Streaming {len(chan_ids)} channels in parallel...\n")
    
    progress_queue = Queue()
    threads = []
    
    for i, cid in enumerate(chan_ids, 1):
        t = threading.Thread(
            target=stream_channel_thread,
            args=(cid, None, metadata, len(chan_ids), i, progress_queue),
            daemon=True
        )
        t.start()
        threads.append(t)
    
    total_windows = progress_monitor(progress_queue, len(chan_ids))
    
    for t in threads:
        t.join(timeout=5.0)
    
    print(f"\n{'='*60}")
    print(f"✅ STREAMING COMPLETE | Total windows: {total_windows}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()