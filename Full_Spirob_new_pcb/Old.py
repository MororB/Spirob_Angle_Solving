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
import glob

# PyQtGraph für Live-Visualisierung
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtCore
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

#.\venv_py312\Scripts\python.exe live_test.py

SERIAL_PORT = "COM6"
BAUD_RATE = 1_000_000
NUM_SENSORS = 6  # Anzahl Sensoren (anpassen falls nötig)
FEATURES_PER_SENSOR = 5  # accX, accY, accZ, magX, magY, magZ

# Performance Settings
SERIAL_READ_SIZE = 2048  # Größerer Buffer für 50Hz (war 256)
MAX_BUFFER_FRAMES = 20   # Verwerfe alte Frames wenn Buffer überläuft
PRINT_EVERY_N = 1       

# === CONFIG: User Settings ===
USE_GPU = True           # True = GPU (falls verfügbar), False = CPU
SIMPLE_OUTPUT = False    # True = einfacher Modus, False = detailliert mit Queue/Frames
VISUALIZATION_MODE = None  # Wird während Runtime gesetzt: "console", "graph", oder None
PREDICTION_GAIN = 1    # Verstärkungsfaktor für Y-Vorhersagen (nur bei MinMaxScaler)

# === Visualisierungs-Parameter ===
MAX_PLOT_POINTS = 200  # Anzahl der angezeigten Punkte im Graph

# === Binary Frame Constants ===
FRAME_HEADER = bytes([0xAA, 0x55])
FRAME_HEADER_SIZE = 2
FRAME_META_SIZE = 9  # frame_id(4) + t_esp_us(4) + num_sensors(1)
SENSOR_PACKET_SIZE = 25  # sensor_id(1) + 6*float32(24)

# === Neural Network Model Definitions ===
class RobotAngleNet(nn.Module):
    """Feedforward Neural Network (New Sequential Architecture)"""
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.1):
        super(RobotAngleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        return self.net(x)

class RobotAngleNetOld(nn.Module):
    """Feedforward Neural Network (Old Architecture - Backward Compatibility)"""
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.1):
        super(RobotAngleNetOld, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # Correct: reduces to half
        self.fc3 = nn.Linear(hidden_size // 2, output_size)  # Correct: from half to output
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class RobotAngleMLP(nn.Module):
    """Flexible MLP with configurable hidden layers (matches nettraining_mlp.ipynb)"""
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

class RobotAngleLSTM(nn.Module):
    """LSTM Recurrent Neural Network"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.1):
        super(RobotAngleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                           dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# === Hilfsfunktion: Neuesten Modell-Ordner finden ===
def get_latest_model_dir(trained_models_dir="trained_models"):
    model_dirs = [d for d in Path(trained_models_dir).iterdir() if d.is_dir()]
    if not model_dirs:
        raise FileNotFoundError("Kein Modellordner gefunden!")
    latest = max(model_dirs, key=lambda d: d.stat().st_mtime)
    return str(latest)

def get_recent_models(trained_models_dir="trained_models", n=5):
    """Gibt die letzten N Modelle zurück, sortiert nach Änderungsdatum (neueste zuerst)"""
    model_dirs = [d for d in Path(trained_models_dir).iterdir() if d.is_dir()]
    if not model_dirs:
        return []
    sorted_dirs = sorted(model_dirs, key=lambda d: d.stat().st_mtime, reverse=True)
    return [str(d) for d in sorted_dirs[:n]]

# === Modell & Scaler laden ===
def load_model_and_scaler(model_dir, device):
    with open(os.path.join(model_dir, "model_config.json"), "r") as f:
        config = json.load(f)
    
    scaler_x = joblib.load(os.path.join(model_dir, "scaler_x.pkl"))
    scaler_y = joblib.load(os.path.join(model_dir, "scaler_y.pkl"))
    input_dim = config["input_size"]
    output_dim = config["output_size"]
    hidden_size = config["hidden_size"]
    dropout_rate = config.get("dropout_rate", 0.1)
    
    # Bestimme Modelltyp aus Dateinamen
    model_name = os.path.basename(model_dir)
    if "LSTM" in model_name:
        num_layers = config.get("num_lstm_layers", 2)
        window_size = config.get("window_size", 10)
        model = RobotAngleLSTM(input_dim, hidden_size, num_layers, output_dim, dropout_rate)
        # Load trained weights
        state_dict = torch.load(os.path.join(model_dir, "model.pth"), map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"   Model Type: LSTM (Window Size: {window_size}, Layers: {num_layers})")
        is_lstm = True
    else:
        # Try loading with new architecture first, fall back to old if needed
        state_dict = torch.load(os.path.join(model_dir, "model.pth"), map_location=device, weights_only=True)
        
        # Check which architecture was used based on state_dict keys
        has_old_keys = any(k.startswith('fc1.') or k.startswith('fc2.') or k.startswith('fc3.') for k in state_dict.keys())
        has_new_keys = any(k.startswith('net.') for k in state_dict.keys())
        
        # Check if config has hidden_layers (new MLP architecture)
        if "hidden_layers" in config and has_new_keys:
            hidden_layers = config["hidden_layers"]
            model = RobotAngleMLP(input_dim, hidden_layers, output_dim, dropout_rate)
            print(f"   Model Type: MLP (Hidden Layers: {hidden_layers})")
        elif has_old_keys and not has_new_keys:
            model = RobotAngleNetOld(input_dim, hidden_size, output_dim, dropout_rate)
            print(f"   Model Type: FeedForward (Old Architecture - fc1/fc2/fc3)")
        else:
            model = RobotAngleNet(input_dim, hidden_size, output_dim, dropout_rate)
            print(f"   Model Type: FeedForward (New Architecture - Sequential)")
        
        window_size = 1
        is_lstm = False
        
        # Now load the state dict into the appropriate architecture
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    feature_cols = config["sensor_features"]
    
    print(f"   Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    return model, scaler_x, scaler_y, feature_cols, config, is_lstm, window_size

# === Sensordaten-Puffer ===
class SensorBuffer:
    def __init__(self, sensor_ids, features_per_sensor):
        self.sensor_ids = sensor_ids
        self.features_per_sensor = features_per_sensor
        self.data = {sid: [0.0]*features_per_sensor for sid in sensor_ids}
        self.last_update_frame = {sid: None for sid in sensor_ids}
        # Cache für berechnete Magnitudes
        self.magnitude_cache = {sid: {'accMag': 0.0, 'magMag': 0.0} for sid in sensor_ids}
        # 🔍 Diagnose: Zähler für fehlende Sensoren
        self.missing_count = {sid: 0 for sid in sensor_ids}
        self.total_frames = 0
    
    def update(self, sensor_id, values, frame_id=None):
        if sensor_id in self.data:
            self.data[sensor_id] = values
            if frame_id is not None:
                self.last_update_frame[sensor_id] = frame_id
            # Berechne Magnitudes einmal beim Update
            acc_mag = np.sqrt(values[0]**2 + values[1]**2 + values[2]**2)
            mag_mag = np.sqrt(values[3]**2 + values[4]**2 + values[5]**2)
            self.magnitude_cache[sensor_id] = {'accMag': float(acc_mag), 'magMag': float(mag_mag)}
    
    def get_diagnostics(self):
        """🔍 Gibt Statistik über fehlende Sensoren zurück"""
        if self.total_frames == 0:
            return "Keine Frames verarbeitet"
        
        result = []
        for sid in sorted(self.sensor_ids):
            missing_pct = (self.missing_count[sid] / self.total_frames) * 100
            if missing_pct > 5:  # Mehr als 5% fehlt
                result.append(f"Sensor {sid}: {missing_pct:.1f}% fehlt")
        
        return "; ".join(result) if result else "Alle Sensoren OK"
    
    def get_feature_vector(self, feature_order, require_frame_id=None):
        # feature_order: Liste von (sensor_id, feature_name)
        # Gibt nur einen Vektor zurück, wenn ALLE Sensoren mindestens einmal aktualisiert wurden
        feature_map = {sid: vals for sid, vals in self.data.items()}
        vec = []
        all_sensors_ready = True
        
        # Extrahiere unique sensor IDs aus feature_order
        unique_sensors = set(sid for sid, _ in feature_order)
        
        # 🔍 Track total frames
        if require_frame_id is not None:
            self.total_frames += 1
        
        # Prüfe ob alle Sensoren vorhanden sind
        for sid in unique_sensors:
            if sid not in feature_map:
                all_sensors_ready = False
                if require_frame_id is not None:
                    self.missing_count[sid] = self.missing_count.get(sid, 0) + 1
                break
            if require_frame_id is not None and self.last_update_frame.get(sid) != require_frame_id:
                all_sensors_ready = False
                if require_frame_id is not None:
                    self.missing_count[sid] = self.missing_count.get(sid, 0) + 1
                break
        
        if not all_sensors_ready:
            return None
        
        # Baue Feature-Vektor mit gecachten Magnitudes (30x schneller!)
        for sid, fname in feature_order:
            if fname == "accMag":
                vec.append(self.magnitude_cache[sid]['accMag'])
            elif fname == "magMag":
                vec.append(self.magnitude_cache[sid]['magMag'])
            else:
                # Normale Feature-Werte
                vals = feature_map[sid]
                idx = ["accX","accY","accZ","magX","magY","magZ"].index(fname)
                vec.append(float(vals[idx]))
        
        return np.array(vec, dtype=np.float32)

# === Binary Frame Parser ===
class BinaryFrameParser:
    def __init__(self, buffer_size=4096):
        self.buffer = bytearray()
        self.buffer_size = buffer_size
    
    def add_data(self, data):
        self.buffer.extend(data)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
    
    def extract_frames(self):
        """Extrahiert komplette Frames aus dem Buffer"""
        frames = []
        while len(self.buffer) >= FRAME_HEADER_SIZE:
            # Suche nach Header
            idx = self.buffer.find(FRAME_HEADER)
            if idx == -1:
                break
            
            if idx > 0:
                self.buffer = self.buffer[idx:]
            
            # Prüfe ob genug Daten für Meta vorhanden
            if len(self.buffer) < FRAME_HEADER_SIZE + FRAME_META_SIZE:
                break
            
            # Parse Frame-Meta
            frame_id, = struct.unpack('<I', self.buffer[2:6])
            t_esp_us, = struct.unpack('<I', self.buffer[6:10])
            num_sensors = self.buffer[10]
            
            # Sanity check
            if num_sensors > 16:
                self.buffer = self.buffer[1:]
                continue
            
            # Prüfe ob kompletter Frame vorhanden
            frame_size = FRAME_HEADER_SIZE + FRAME_META_SIZE + num_sensors * SENSOR_PACKET_SIZE
            if len(self.buffer) < frame_size:
                break
            
            # Extrahiere Frame
            frame_data = self.buffer[:frame_size]
            self.buffer = self.buffer[frame_size:]
            
            # Parse Sensor-Pakete
            sensor_data = {}
            for i in range(num_sensors):
                offset = FRAME_HEADER_SIZE + FRAME_META_SIZE + i * SENSOR_PACKET_SIZE
                sensor_id = frame_data[offset]
                values = struct.unpack('<6f', frame_data[offset+1:offset+25])
                sensor_data[sensor_id] = values
            
            frames.append({
                'frame_id': frame_id,
                't_esp_us': t_esp_us,
                'num_sensors': num_sensors,
                'sensors': sensor_data
            })
        
        return frames

# === Feature-Order aus Trainingskonfig bauen ===
def build_feature_order(feature_cols):
    # feature_cols z.B. ['accX_0', 'accY_0', ..., 'magZ_12']
    # Extrahiere (sensor_id, feature_name) Paare
    order = []
    for col in feature_cols:
        # Format: "featureName_sensorId" z.B. "accX_0"
        parts = col.rsplit('_', 1)  # Split vom rechten Ende
        if len(parts) == 2:
            feature_name = parts[0]  # "accX"
            try:
                sensor_id = int(parts[1])  # 0
                order.append((sensor_id, feature_name))
            except ValueError:
                print(f"[WARNING] Konnte Feature-Spalte nicht parsen: {col}")
        else:
            print(f"[WARNING] Unbekannte Feature-Spalte: {col}")
    
    return order

# === Live-Test Hauptfunktion ===
# === Serial Reader Thread ===
class SerialReader(threading.Thread):
    def __init__(self, port, baud_rate):
        super().__init__(daemon=True)
        self.port = port
        self.baud_rate = baud_rate
        self.running = False
        self.frame_queue = deque(maxlen=MAX_BUFFER_FRAMES)
        self.parser = BinaryFrameParser(buffer_size=8192)
        
    def run(self):
        self.running = True
        try:
            print(f"Serial Thread gestartet: {self.port} @ {self.baud_rate}")
            ser = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            ser.reset_input_buffer()
            
            while self.running:
                if ser.in_waiting:
                    data = ser.read(ser.in_waiting)
                    self.parser.add_data(data)
                    frames = self.parser.extract_frames()
                    for frame in frames:
                        self.frame_queue.append(frame)
                else:
                    time.sleep(0.001)
                    
        except Exception as e:
            print(f"\nSerial Error im Thread: {e}")
            self.running = False
        finally:
            if 'ser' in locals() and ser.is_open:
                ser.close()

# === Live-Test Hauptfunktion ===
def main():
    # GPU Setup (automatisch basierend auf Config)
    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n✅ Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"\n✅ Device: CPU")
         
    # Modell-Auswahl: Zeige letzte 5 Modelle
    print(f"\n📁 Verfügbare Modelle:")
    recent_models = get_recent_models("trained_models", n=5)
    
    if not recent_models:
        print("   Keine Modelle gefunden!"); return
    
    for i, model_path in enumerate(recent_models, 1):
        model_name = os.path.basename(model_path)
        print(f"   [{i}] {model_name}")
    
    print(f"\n   [Enter] = Neustes Modell (1)")
    user = input("Modell wählen (1-5 oder Enter): ").strip()
    
    if user == "":
        model_dir = recent_models[0]
        print(f"→ Nutze neustes Modell: {os.path.basename(model_dir)}")
    elif user.isdigit() and 1 <= int(user) <= len(recent_models):
        model_dir = recent_models[int(user) - 1]
        print(f"→ Nutze Modell: {os.path.basename(model_dir)}")
    else:
        print(f"Ungültige Auswahl!"); return

    model, scaler_x, scaler_y, feature_cols, config, is_lstm, window_size = load_model_and_scaler(model_dir, device)
    feature_order = build_feature_order(feature_cols)
    sensor_ids = sorted(set(sid for sid, _ in feature_order))
    expected_sensor_ids = set(sensor_ids)
    buffer = SensorBuffer(sensor_ids, FEATURES_PER_SENSOR)
    
    # 🔍 DEBUG: Print expected features
    print(f"\n📋 Feature-Konfiguration:")
    print(f"   Erwartet {len(feature_cols)} Features: {feature_cols}")
    print(f"   Feature-Order (sensor_id, feature_name):")
    for i, (sid, fname) in enumerate(feature_order[:min(16, len(feature_order))]):
        print(f"      [{i:2d}] Sensor {sid:2d}: {fname}")
    if len(feature_order) > 16:
        print(f"      ... ({len(feature_order)-16} weitere)")
    print(f"   Sensoren gesamt: {sensor_ids}")
    
    # Zeige Scaler-Info
    if isinstance(scaler_x, list):
        print(f"   Scaler X: Liste von {len(scaler_x)} Scalern (individuell pro Sensor)")
        print(f"   Scaler X[0] type: {type(scaler_x[0]).__name__}")
        if hasattr(scaler_x[0], "center_"):
            print(f"   Scaler X[0] center: {scaler_x[0].center_[:3]}... (erste 3 Features)")
    else:
        print(f"   Scaler X type: {type(scaler_x).__name__}")
        if hasattr(scaler_x, "mean_"):
            print(f"   Scaler X mean: {scaler_x.mean_[:5]}... (erste 5 Features)")
        elif hasattr(scaler_x, "center_"):
            print(f"   Scaler X center: {scaler_x.center_[:5]}... (erste 5 Features)")
        if hasattr(scaler_x, "scale_"):
            print(f"   Scaler X scale: {scaler_x.scale_[:5]}... (erste 5 Features)")
        elif hasattr(scaler_x, "data_range_"):
            print(f"   Scaler X range: {scaler_x.data_range_[:5]}... (erste 5 Features)")
    
    # LSTM benötigt einen History-Buffer
    if is_lstm:
        history_buffer = deque(maxlen=window_size)
    
    # 🚀 Performance: Pre-allocate tensors für Reuse (verhindert wiederholte GPU-Transfers)
    if is_lstm:
        input_tensor = torch.zeros((1, window_size, len(feature_cols)), dtype=torch.float32, device=device)
    else:
        input_tensor = torch.zeros((1, len(feature_cols)), dtype=torch.float32, device=device)
    
    # Start Serial Thread
    reader = SerialReader(SERIAL_PORT, BAUD_RATE)
    reader.start()

    print("\n" + "="*60)
    print("--- LIVE TEST (Multithreaded) --- (Strg+C zum Beenden)")
    print("="*60)
    
    # Performance tracking
    inference_times = deque(maxlen=100)
    total_frames_processed = 0
    last_print_time = time.time()
    missing_frames = 0
    last_missing_report = time.time()
    last_missing_ids = []
    
    # Output-Modus aus Config
    simple_mode = SIMPLE_OUTPUT
    
    # 🎯 Smoothing-Option für stabilere Winkel
    use_smoothing = input("\nWinkel-Smoothing aktivieren? (j/n, Enter=j): ").strip().lower()
    use_smoothing = use_smoothing != 'n'  # Default: ja
    smoothing_window = 5 if use_smoothing else 1
    
    if use_smoothing:
        print(f"\n✅ Smoothing aktiviert (Fenster: {smoothing_window} Frames)")
        angle_history = deque(maxlen=smoothing_window)
    else:
        print("\n⚠️  Smoothing deaktiviert - zeigt rohe Vorhersagen")
    
    # 📊 Visualisierungs-Modus wählen
    viz_mode = input("\nWinkel-Ausgabe (c=console/g=graph, Enter=console): ").strip().lower()
    
    if viz_mode in ["graph", "g"]:
        if not HAS_PYQTGRAPH:
            print("❌ PyQtGraph nicht verfügbar. Fallback auf Console-Modus.")
            VISUALIZATION_MODE = "console"
            angle_buffer = None
            visualizer_app = None
        else:
            VISUALIZATION_MODE = "graph"
            print("✅ Graph-Modus aktiviert (PyQtGraph)")
            
            # PyQtGraph Setup
            pg.setConfigOptions(antialias=True, useOpenGL=True)
            
            num_angles = len(config["output_columns"])
            angle_buffer = deque(maxlen=MAX_PLOT_POINTS)
            
            # Farben für verschiedene Gelenke
            color_palette = [
                (255, 100, 100),   # Rot
                (100, 255, 100),   # Grün
                (100, 100, 255),   # Blau
                (255, 255, 100),   # Gelb
                (255, 100, 255),   # Magenta
                (100, 255, 255),   # Cyan
                (255, 180, 100),   # Orange
                (180, 100, 255),   # Violett
            ]
            
            # Erstelle App und Fenster
            visualizer_app = QtWidgets.QApplication.instance()
            if visualizer_app is None:
                visualizer_app = QtWidgets.QApplication([])
            
            win = pg.GraphicsLayoutWidget(title="Live Angle Visualization")
            win.resize(1200, 600)
            
            # Plot für Winkel
            plot_angles = win.addPlot(title="Joint Angles (θ₁ - θ₁₃)")
            plot_angles.setLabel('left', 'Angle', units='°')
            plot_angles.setLabel('bottom', 'Sample Index')
            plot_angles.setYRange(-50, 50)
            plot_angles.showGrid(x=True, y=True, alpha=0.3)
            plot_angles.addLegend()
            
            # Kurven für jeden Winkel
            curves = []
            x_data = np.arange(MAX_PLOT_POINTS)
            
            for i in range(num_angles):
                color = color_palette[i % len(color_palette)]
                pen = pg.mkPen(color=color, width=2)
                curve = plot_angles.plot(name=f'θ{i+1}', pen=pen)
                curves.append(curve)
            
            win.show()
            plot_update_timer = QtCore.QTimer()
            
            # Update-Funktion für PyQtGraph
            def update_plot():
                if len(angle_buffer) > 0:
                    angle_array = np.array(list(angle_buffer))  # (N_samples, N_angles)
                    for i, curve in enumerate(curves):
                        curve.setData(x_data[:len(angle_buffer)], angle_array[:, i])
            
            plot_update_timer.timeout.connect(update_plot)
            plot_update_timer.start(100)  # 10Hz Update
    else:
        VISUALIZATION_MODE = "console"
        angle_buffer = None
        visualizer_app = None
        print("✅ Console-Modus aktiviert")
    
    try:
        while reader.is_alive():
            # Qt Event Processing für Graph-Modus
            if VISUALIZATION_MODE == "graph":
                visualizer_app.processEvents()
            
            # Hole ALLE verfügbaren Frames aus der Queue
            current_frames = []
            while reader.frame_queue:
                try:
                    current_frames.append(reader.frame_queue.popleft())
                except IndexError:
                    break
            
            if not current_frames:
                time.sleep(0.005) # Kurzer Sleep wenn keine Daten (verhindert CPU-Last)
                continue
                
            # Update state for ALL frames
            do_inference = False
            frame_sensor_data = {}
            
            for frame in current_frames:
                # Aktualisiere Sensoren
                for sensor_id, values in frame['sensors'].items():
                    buffer.update(sensor_id, values, frame_id=frame['frame_id'])
                    frame_sensor_data[sensor_id] = values
                missing = expected_sensor_ids - set(frame['sensors'].keys())
                if missing:
                    missing_frames += 1
                    last_missing_ids = sorted(missing)
                    if time.time() - last_missing_report > 1.0:
                        print(f"\n⚠️  Frame {frame['frame_id']}: Missing sensors {last_missing_ids} (have {sorted(frame['sensors'].keys())})", flush=True)
                        last_missing_report = time.time()
                    continue
                
                # Feature-Vektor bauen
                x_vec = buffer.get_feature_vector(feature_order, require_frame_id=frame['frame_id'])
                if x_vec is None:
                    continue

                # 🆕 Skalierung: Unterstütze sowohl einzelnen Scaler als auch Liste von Scalern
                if isinstance(scaler_x, list):
                    # Individuelle Scaler pro Sensor
                    x_scaled = np.zeros_like(x_vec)
                    features_per_sensor = len(x_vec) // len(scaler_x)
                    for i, scaler in enumerate(scaler_x):
                        start_idx = i * features_per_sensor
                        end_idx = start_idx + features_per_sensor
                        x_scaled[start_idx:end_idx] = scaler.transform([x_vec[start_idx:end_idx]])[0]
                else:
                    # Einzelner Scaler für alle Features (altes Format)
                    x_scaled = scaler_x.transform([x_vec])[0]
                
                if is_lstm:
                    history_buffer.append(x_scaled)
                    if len(history_buffer) < window_size: continue
                    if frame == current_frames[-1]:
                        # 🚀 Update existing tensor statt neu zu erstellen (10x schneller!)
                        input_tensor[0] = torch.from_numpy(np.array(list(history_buffer), dtype=np.float32))
                        do_inference = True
                    else:
                        do_inference = False
                else:
                    if frame == current_frames[-1]:
                        # 🚀 Update existing tensor statt neu zu erstellen
                        input_tensor[0] = torch.from_numpy(x_scaled.astype(np.float32))
                        do_inference = True
                    else:
                        do_inference = False

            total_frames_processed += len(current_frames)

            # Nur inferieren wenn ready
            if do_inference:
                t_start = time.perf_counter()
                with torch.no_grad():
                    y_pred_scaled = model(input_tensor).cpu().numpy()[0]
                    # Rücktransformation (scaler_y kann None sein oder MinMaxScaler)
                    if scaler_y is not None:
                        y_pred_raw = scaler_y.inverse_transform([y_pred_scaled * PREDICTION_GAIN])[0]
                    else:
                        y_pred_raw = y_pred_scaled  # Modell gibt direkt Grad aus
                
                # 🎯 Smoothing anwenden
                if use_smoothing:
                    angle_history.append(y_pred_raw)
                    if len(angle_history) >= 2:
                        y_pred = np.mean(angle_history, axis=0)
                    else:
                        y_pred = y_pred_raw
                else:
                    y_pred = y_pred_raw
                
                inference_time = (time.perf_counter() - t_start) * 1000
                inference_times.append(inference_time)
                
                # 📊 Graph-Buffer SOFORT füllen (nicht auf Print-Timer warten!)
                if VISUALIZATION_MODE == "graph":
                    angle_buffer.append(y_pred)
                
                # Console-Ausgabe bei jeder Inferenz
                avg_inf = np.mean(inference_times) if inference_times else 0
                
                if VISUALIZATION_MODE != "graph":
                    if simple_mode:
                        # 🚀 Einfacher Modus (wie alter Code - minimal Overhead)
                        angles_str = ' '.join(f"{a:6.1f}" for a in y_pred)
                        print(f"\rWinkel: [{angles_str}]  ({avg_inf:.1f}ms)", end="", flush=True)
                    else:
                        # Detaillierter Modus
                        angles_str = ' '.join(f"{a:5.1f}" for a in y_pred)
                        print(f"Q:{len(reader.frame_queue):3d} | Total:{total_frames_processed:6d} | {avg_inf:3.1f}ms | θ:[{angles_str}]", end="\r", flush=True)

                if time.time() - last_missing_report > 2.0 and missing_frames > 0:
                    last_missing_report = time.time()
                    print(f"\n⚠️  Missing sensors in {missing_frames} frame(s). Last missing IDs: {last_missing_ids}")
                    missing_frames = 0
                
                # 🔍 Diagnose alle 5 Sekunden
                if time.time() - last_print_time > 5.0:
                    diag = buffer.get_diagnostics()
                    if "fehlt" in diag:
                        print(f"\n🔍 Sensor-Diagnose: {diag}", flush=True)

    except KeyboardInterrupt:
        print("\nStop...")
    finally:
        reader.running = False
        reader.join(timeout=1.0)
        
        # Cleanup für PyQtGraph
        if VISUALIZATION_MODE == "graph" and visualizer_app is not None:
            try:
                plot_update_timer.stop()
                win.close()
                visualizer_app.quit()
            except:
                pass
        
        print("Done.")

if __name__ == "__main__":
    main()
