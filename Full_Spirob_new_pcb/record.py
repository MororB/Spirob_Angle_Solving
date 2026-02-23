import serial
import time
import csv
import threading
import cv2
import numpy as np
import sys
from pathlib import Path
import queue
import struct

# ===================== CONFIGURATION =====================
SERIAL_PORT = "COM6"  # Updated to match Sensor_Visualizer.py (was COM5 in old record.py)
BAUD_RATE = 1000000
OUTPUT_DIR = Path("runs")
SYNC_INTERVAL_MS = 1000  # Change visual sync state every 1000ms (1 second)

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)
RUN_TIMESTAMP = time.strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = OUTPUT_DIR / RUN_TIMESTAMP
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Files
SENSOR_CSV = RUN_DIR / "sensors.csv"
SYNC_CSV = RUN_DIR / "sync_events.csv"

# Global flags
running = True

# ===================== SERIAL LOGGER =====================
FRAME_HDR_0 = 0xAA
FRAME_HDR_1 = 0x55

# Binary layout: header(2) + frame_id(uint32) + t_us(uint32) + n(uint8) + n * (sensor_id(uint8) + 6 floats)
FRAME_FIXED_SIZE = 2 + 4 + 4 + 1
SENSOR_PACKET_SIZE = 1 + 6 * 4

def read_exact(ser, n, timeout_sec=0.1):
    buf = bytearray()
    t_start = time.time()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if chunk:
            buf.extend(chunk)
        if (time.time() - t_start) > timeout_sec:
            return None
    return bytes(buf)
def serial_logging_thread(serial_port, baud_rate, output_file):
    global running
    
    print(f"[Serial] Connecting to {serial_port} @ {baud_rate}...")
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=0.1)
        ser.reset_input_buffer()
        print(f"[Serial] Connected! Logging to {output_file}...")
    except Exception as e:
        print(f"[Serial] Error opening port: {e}")
        running = False
        return

    # Buffer for batch writing to avoid IO bottlenecks
    batch_data = []
    last_flush = time.time()
    frame_count = 0
    
    # Build initial buffer and sync to first frame header
    print("[Serial] Syncing to first frame header...")
    buffer = bytearray()
    synced = False
    
    while not synced and running:
        if ser.in_waiting:
            buffer.extend(ser.read(ser.in_waiting))
        
        # Look for header in buffer
        idx = buffer.find(bytes([FRAME_HDR_0, FRAME_HDR_1]))
        if idx >= 0:
            # Found header, discard everything before it
            buffer = buffer[idx:]
            synced = True
            print(f"[Serial] Synced! Discarded {idx} bytes, buffer starts with header")
        elif len(buffer) > 10000:
            # Too much garbage, reset
            print("[Serial] Warning: 10KB without header, resetting buffer")
            buffer = buffer[-100:]
        
        if not synced:
            time.sleep(0.01)
    
    if not synced:
        print("[Serial] Failed to sync to frame header")
        return
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header includes ESP frame_id (no ESP timestamp stored)
        writer.writerow(["t_pc_ns", "frame_id", "sensor_id", "accX", "accY", "accZ", "magX", "magY", "magZ"])
        
        while running:
            try:
                # Read more data into buffer
                if ser.in_waiting:
                    buffer.extend(ser.read(ser.in_waiting))
                else:
                    time.sleep(0.001)
                
                # Process all complete frames in buffer
                while len(buffer) >= 2:
                    # Look for header
                    if buffer[0] != FRAME_HDR_0 or buffer[1] != FRAME_HDR_1:
                        # Not a header, search for next one
                        idx = buffer.find(bytes([FRAME_HDR_0, FRAME_HDR_1]), 1)
                        if idx > 0:
                            buffer = buffer[idx:]
                        else:
                            buffer = buffer[-1:]
                        continue
                    
                    # Check if we have enough bytes for header
                    if len(buffer) < FRAME_FIXED_SIZE:
                        break
                    
                    # Parse header
                    frame_id, _t_esp_us, n = struct.unpack_from("<IIB", buffer, 2)
                    
                    # Sanity check
                    if n == 0 or n > 16:
                        buffer = buffer[2:]
                        continue
                    
                    total_len = FRAME_FIXED_SIZE + n * SENSOR_PACKET_SIZE
                    if len(buffer) < total_len:
                        break
                    
                    # Extract frame
                    payload = buffer[FRAME_FIXED_SIZE:total_len]
                    buffer = buffer[total_len:]
                    
                    t_ns = time.perf_counter_ns()
                    offset = 0
                    for _ in range(n):
                        pkt = payload[offset:offset + SENSOR_PACKET_SIZE]
                        sensor_id = pkt[0]
                        accX, accY, accZ, magX, magY, magZ = struct.unpack("<ffffff", pkt[1:])
                        row = [t_ns, frame_id, sensor_id, accX, accY, accZ, magX, magY, magZ]
                        batch_data.append(row)
                        offset += SENSOR_PACKET_SIZE
                    
                    frame_count += 1
                    if frame_count % 100 == 0:
                        print(f"[Serial] Received {frame_count} frames ({n} sensors/frame), {len(batch_data)} samples buffered")
                
                # Periodic flush
                if len(batch_data) > 100 or (time.time() - last_flush > 0.1):
                    if batch_data:
                        writer.writerows(batch_data)
                        f.flush()
                        batch_data = []
                    last_flush = time.time()
                    
            except Exception as e:
                print(f"[Serial] Read Error: {e}")
                break
        
        # Final flush before exit
        if batch_data:
            writer.writerows(batch_data)
            f.flush()
            print(f"[Serial] Flushed final {len(batch_data)} samples")
        
        print(f"[Serial] Total frames received: {frame_count}")
                
        ser.close()
        print("[Serial] Port closed.")

# ===================== SYNC VISUALIZER =====================
def main():
    global running
    
    # 1. Start Serial Thread
    t_serial = threading.Thread(target=serial_logging_thread, args=(SERIAL_PORT, BAUD_RATE, SENSOR_CSV))
    t_serial.daemon = False  # Changed to False so thread finishes properly
    t_serial.start()
    
    # 2. Setup Sync Window
    window_name = "SYNC TRACKER - Keep Visible in Camera!"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    # Sync Logic
    current_marker_id = 100
    first_run = True
    last_switch_time = 0
    
    # ArUco Configuration (4x4 Dictionary, 1000 to allow IDs > 50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    
    print(f"[Main] Starting Sync Loop. Press ESC to stop.")
    
    with open(SYNC_CSV, 'w', newline='') as f_sync:
        writer_sync = csv.writer(f_sync)
        writer_sync.writerow(["t_pc_ns", "sync_id", "state", "description"])
        
        while running:
            current_time = time.time() * 1000 # ms
            
            # Helper to get high precision time for logging
            now_ns = time.perf_counter_ns()
            
            # --- Update Sync State Periodically ---
            if current_time - last_switch_time > SYNC_INTERVAL_MS:
                if first_run:
                    first_run = False
                else:
                    current_marker_id += 1
                
                last_switch_time = current_time
                
                # Log the switch event
                writer_sync.writerow([now_ns, current_marker_id, "MARKER_UPDATE", f"ArUco ID {current_marker_id}"])
                f_sync.flush()

            # --- Draw UI ---
            # Create white background image
            img = np.ones((600, 800, 3), dtype=np.uint8) * 255
            
            try:
                # Generate ArUco Marker
                # size 400x400
                marker_img = cv2.aruco.generateImageMarker(aruco_dict, current_marker_id, 400)
                marker_img_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
                
                # Center the marker
                y_off = (600 - 400) // 2
                x_off = (800 - 400) // 2
                
                img[y_off:y_off+400, x_off:x_off+400] = marker_img_bgr
                
            except Exception as e:
                # Fallback if generation fails (e.g. ID out of range)
                cv2.putText(img, f"Error: {e}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # Draw Text info
            cv2.putText(img, f"SYNC ID: {current_marker_id}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Progress Bar (Red line at bottom)
            cw = 800
            progress = int(((current_time - last_switch_time) / SYNC_INTERVAL_MS) * cw)
            cv2.rectangle(img, (0, 580), (progress, 600), (0, 0, 255), -1)
            
            cv2.imshow(window_name, img)
            
            key = cv2.waitKey(10) & 0xFF
            if key == 27: # ESC
                running = False
                break
    
    # Cleanup
    print("[Main] Stopping...")
    running = False
    t_serial.join(timeout=2.0)  # Wait for thread to finish
    cv2.destroyAllWindows()
    
    print(f"[Done] Recordings saved to {RUN_DIR}")
    
    # Meta data
    meta_path = RUN_DIR / "recording_info.txt"
    with open(meta_path, 'w') as f:
        f.write(f"Run: {RUN_TIMESTAMP}\n")
        f.write(f"Serial: {SERIAL_PORT} @ {BAUD_RATE}\n")
        f.write(f"Columns: DATA,ID,accX,accY,accZ,magX,magY,magZ\n")
        f.write(f"Sync: Toggles every {SYNC_INTERVAL_MS}ms\n")
        
    print(f"[Done] Recordings saved to {RUN_DIR}")

if __name__ == "__main__":
    main()