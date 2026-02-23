import serial
import struct
import threading
import time
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import json
from collections import deque
from pathlib import Path

# PyQtGraph for Live-Visualization
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtCore
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

# === CONFIGURATION ===
SERIAL_PORT = "COM6"
BAUD_RATE = 1_000_000
MAX_BUFFER_FRAMES = 20
USE_GPU = True
PREDICTION_GAIN = 1.0

# Binary Frame Constants
FRAME_HEADER = bytes([0xAA, 0x55])
FRAME_HEADER_SIZE = 2
FRAME_META_SIZE = 9  # frame_id(4) + t_esp_us(4) + num_sensors(1)
SENSOR_PACKET_SIZE = 25  # sensor_id(1) + 6*float32(24)

class RobotAngleMLP(nn.Module):
    """Flexible MLP architecture matching nettraining_mlp.ipynb"""
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate=0.2):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def get_recent_models(trained_models_dir="trained_models", n=5):
    """Returns the last N models sorted by modification time"""
    if not os.path.exists(trained_models_dir): return []
    model_dirs = [d for d in Path(trained_models_dir).iterdir() if d.is_dir()]
    sorted_dirs = sorted(model_dirs, key=lambda d: d.stat().st_mtime, reverse=True)
    return [str(d) for d in sorted_dirs[:n]]

def load_model_and_scaler(model_dir, device):
    """Loads weights, config, and scalers from a model directory"""
    with open(os.path.join(model_dir, "model_config.json"), "r") as f:
        config = json.load(f)
    
    scaler_x = joblib.load(os.path.join(model_dir, "scaler_x.pkl"))
    scaler_y = joblib.load(os.path.join(model_dir, "scaler_y.pkl"))
    
    hidden_layers = config.get("hidden_layers", [256, 128, 64])
    input_dim = config["input_size"]
    output_dim = config["output_size"]
    dropout = config.get("dropout_rate", 0.1)
    
    model = RobotAngleMLP(input_dim, hidden_layers, output_dim, dropout)
    state_dict = torch.load(os.path.join(model_dir, "model.pth"), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"   Model: {os.path.basename(model_dir)}")
    print(f"   Architecture: {input_dim} -> {hidden_layers} -> {output_dim}")
    return model, scaler_x, scaler_y, config["sensor_features"], config

class SensorBuffer:
    """Buffers and synchronizes packets from multiple sensors"""
    def __init__(self, sensor_ids):
        self.sensor_ids = sensor_ids
        self.data = {sid: [0.0]*6 for sid in sensor_ids}
        self.last_update_id = {sid: None for sid in sensor_ids}
    
    def update(self, sensor_id, values, frame_id):
        if sensor_id in self.data:
            self.data[sensor_id] = values
            self.last_update_id[sensor_id] = frame_id
    
    def get_feature_vector(self, feature_order, frame_id):
        # Extract unique sensors needed for the feature vector
        unique_needed = set(sid for sid, _ in feature_order)
        
        # Ensure all needed sensors have data for THIS specific frame_id
        for sid in unique_needed:
            if sid not in self.data or self.last_update_id.get(sid) != frame_id:
                return None
        
        vec = []
        for sid, fname in feature_order:
            vals = self.data[sid]
            idx = ["accX","accY","accZ","magX","magY","magZ"].index(fname)
            vec.append(float(vals[idx]))
        
        return np.array(vec, dtype=np.float32)

class BinaryFrameParser:
    """Parses the binary ESP32 sensor stream"""
    def __init__(self, buffer_size=8192):
        self.buffer = bytearray()
        self.buffer_size = buffer_size
    
    def add_data(self, data):
        self.buffer.extend(data)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
    
    def extract_frames(self):
        frames = []
        while len(self.buffer) >= FRAME_HEADER_SIZE + FRAME_META_SIZE:
            idx = self.buffer.find(FRAME_HEADER)
            if idx == -1: break
            if idx > 0: self.buffer = self.buffer[idx:]
            
            if len(self.buffer) < FRAME_HEADER_SIZE + FRAME_META_SIZE: break
            
            frame_id, t_esp_us, num_sensors = struct.unpack('<IIB', self.buffer[2:11])
            frame_size = FRAME_HEADER_SIZE + FRAME_META_SIZE + num_sensors * SENSOR_PACKET_SIZE
            
            if len(self.buffer) < frame_size: break
            
            frame_bytes = self.buffer[:frame_size]
            self.buffer = self.buffer[frame_size:]
            
            sensors = {}
            for i in range(num_sensors):
                offset = 11 + i * SENSOR_PACKET_SIZE
                sid = frame_bytes[offset]
                vals = struct.unpack('<6f', frame_bytes[offset+1:offset+25])
                sensors[sid] = vals
            
            frames.append({'id': frame_id, 'sensors': sensors})
        return frames

class SerialReader(threading.Thread):
    """Background thread for continuous serial data reading"""
    def __init__(self, port, baud_rate):
        super().__init__(daemon=True)
        self.port, self.baud_rate = port, baud_rate
        self.running = False
        self.frame_queue = deque(maxlen=MAX_BUFFER_FRAMES)
        self.parser = BinaryFrameParser()
        
    def run(self):
        self.running = True
        try:
            ser = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            ser.reset_input_buffer()
            while self.running:
                if ser.in_waiting:
                    self.parser.add_data(ser.read(ser.in_waiting))
                    for frame in self.parser.extract_frames():
                        self.frame_queue.append(frame)
                else: time.sleep(0.001)
        except Exception as e:
            print(f"Serial Error: {e}"); self.running = False

def build_feature_order(feature_cols):
    """Parses (sensor_id, feature_name) from training column headers"""
    order = []
    for col in feature_cols:
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            order.append((int(parts[1]), parts[0]))
    return order

def main():
    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    print(f"\n✅ Device: {device}")

    # Model Selection
    models = get_recent_models()
    if not models: print("No models found!"); return
    for i, m in enumerate(models, 1): print(f"  [{i}] {os.path.basename(m)}")
    
    choice = input("\nSelect Model [Enter=1]: ").strip()
    model_dir = models[int(choice)-1 if choice.isdigit() else 0]
    
    model, scaler_x, scaler_y, f_cols, config = load_model_and_scaler(model_dir, device)
    feature_order = build_feature_order(f_cols)
    buffer = SensorBuffer(sorted(set(sid for sid, _ in feature_order)))
    
    # Visualization and Smoothing
    viz = input("\nMode: [c]onsole, [g]raph (Enter=c): ").strip().lower()
    mode = "graph" if viz == 'g' and HAS_PYQTGRAPH else "console"
    
    smooth_in = input("Smoothing Window [Enter=5, 1=None]: ").strip()
    window_size = int(smooth_in) if smooth_in.isdigit() else 5
    smooth_buffer = deque(maxlen=window_size) if window_size > 1 else None

    # PyQtGraph Setup
    app, curves = None, []
    if mode == "graph":
        app = QtWidgets.QApplication([])
        win = pg.GraphicsLayoutWidget(title="Live Angles")
        plot = win.addPlot(title="Joint Angles")
        plot.setYRange(-45, 45)
        plot.addLegend()
        history = deque(maxlen=200)
        curves = [plot.plot(name=f"theta_{i+1}", pen=pg.mkPen(i, width=2)) for i in range(len(config["output_columns"]))]
        win.show()

    reader = SerialReader(SERIAL_PORT, BAUD_RATE)
    reader.start()

    try:
        print(f"\nRunning with Smoothing={window_size}... (Ctrl+C to stop)")
        while reader.is_alive():
            if mode == "graph": app.processEvents()
            
            while reader.frame_queue:
                frame = reader.frame_queue.popleft()
                for sid, vals in frame['sensors'].items():
                    buffer.update(sid, vals, frame['id'])
                
                x_raw = buffer.get_feature_vector(feature_order, frame['id'])
                if x_raw is not None:
                    # Scaling
                    if isinstance(scaler_x, list):
                        x_scaled = np.zeros_like(x_raw)
                        fps = len(x_raw) // len(scaler_x)
                        for i, s in enumerate(scaler_x):
                            x_scaled[i*fps:(i+1)*fps] = s.transform([x_raw[i*fps:(i+1)*fps]])[0]
                    else: x_scaled = scaler_x.transform([x_raw])[0]
                    
                    # Inference
                    with torch.no_grad():
                        t_inp = torch.from_numpy(x_scaled.astype(np.float32)).to(device).unsqueeze(0)
                        y_pred = model(t_inp).cpu().numpy()[0]
                    
                    # Inverse Scale to degrees
                    y_deg_raw = scaler_y.inverse_transform([y_pred * PREDICTION_GAIN])[0]
                    
                    # Smoothing
                    if smooth_buffer is not None:
                        smooth_buffer.append(y_deg_raw)
                        y_deg = np.mean(smooth_buffer, axis=0)
                    else:
                        y_deg = y_deg_raw
                    
                    if mode == "graph":
                        history.append(y_deg)
                        arr = np.array(list(history))
                        for i, c in enumerate(curves): c.setData(arr[:, i])
                    else:
                        print(f"\rAngles: [{', '.join(f'{a:5.1f}' for a in y_deg)}]", end="", flush=True)
            time.sleep(0.001)

    except KeyboardInterrupt: print("\nStopped.")
    finally: reader.running = False

if __name__ == "__main__":
    main()
