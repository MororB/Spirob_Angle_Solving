import time
import struct
import threading
import serial
import numpy as np
import torch
import torch.nn as nn
import joblib
import math
import os
import json
from collections import deque

# ================= KONFIGURATION =================
SERIAL_PORT = "COM6"
BAUD_RATE = 1_000_000

# HIER DEINEN NEUEN ORDNER EINTRAGEN (Der mit "SensX" im Namen)
#RUN_FOLDER_NAME = "2025-12-15_12-47-11_LSTM_h64_Sens3" 
#UN_FOLDER_NAME = "2025-12-15_16-28-41_LSTM_h64_Sens1"
RUN_FOLDER_NAME = "2026-01-19_16-15-57_LSTM_h64_Sens1"

# Hardware Setup (Bleibt immer gleich, wir filtern erst im Brain)
NUM_BOARDS = 3
NUM_SENSORS_PER_BOARD = 5
NUM_FEATURES_PER_SENSOR = 4  # x, y, z, mag

# ================= MODELL =================
class SegmentLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SegmentLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

# ================= KI GEHIRN (MIT FILTER) =================
class NeuralBrain:
    def __init__(self, run_folder_name):
        self.device = torch.device("cpu")
        base_path = os.path.join("./trained_models", run_folder_name)
        
        print(f"--- Lade Brain: {run_folder_name} ---")
        
        # 1. Config laden
        config_path = os.path.join(base_path, "model_config.json")
        with open(config_path, "r") as f:
            self.conf = json.load(f)
            
        # Parameter
        self.hidden_size = self.conf['architecture']['hidden_size']
        self.num_layers = self.conf['architecture']['num_layers']
        try:
            self.window_size = self.conf['training_params']['window_size']
        except:
            self.window_size = 10
        
        # SENSOR AUSWAHL LADEN
        # Fallback auf [0,1,2,3,4] falls altes Modell ohne diese Info
        self.active_sensors = self.conf.get('active_sensors', [0, 1, 2, 3, 4])
        print(f"-> Nutze Sensoren pro Board: {self.active_sensors}")
        
        # 2. Input-Maske berechnen
        # Wir müssen wissen, an welchen Stellen im großen Array (60 floats)
        # unsere gewünschten Daten liegen.
        self.input_indices = []
        for b in range(NUM_BOARDS):
            for s in range(NUM_SENSORS_PER_BOARD):
                if s in self.active_sensors:
                    # Der SerialReader schreibt interleaved: [S0x, S0y, S0z, S0mag, S1x ...]
                    # Index-Berechnung:
                    global_sensor_idx = b * NUM_SENSORS_PER_BOARD + s
                    start_idx = global_sensor_idx * NUM_FEATURES_PER_SENSOR
                    
                    # Wir brauchen alle 4 Features (x,y,z,mag) dieses Sensors
                    self.input_indices.extend([start_idx, start_idx+1, start_idx+2, start_idx+3])
        
        self.input_indices = np.array(self.input_indices, dtype=np.int32)
        
        # 3. Scaler & Modell
        self.scaler_x = joblib.load(os.path.join(base_path, "scaler_x.pkl"))
        
        # Scaler Y Check
        scaler_y_path = os.path.join(base_path, "scaler_y.pkl")
        if os.path.exists(scaler_y_path):
            self.scaler_y = joblib.load(scaler_y_path)
            self.use_scaler_y = True
        else:
            self.use_scaler_y = False
            self.angle_max = self.conf.get('manual_angle_scale', 60.0)

        # Netz laden
        self.input_dim = self.scaler_x.n_features_in_ # Sollte zur Länge von input_indices passen
        if len(self.input_indices) != self.input_dim:
            print(f"WARNUNG: Berechnete Indizes ({len(self.input_indices)}) passen nicht zum Scaler ({self.input_dim})!")
        
        self.model = SegmentLSTM(self.input_dim, self.hidden_size, self.num_layers, 4)
        self.model.load_state_dict(torch.load(os.path.join(base_path, "model.pth"), map_location=self.device))
        self.model.eval()
        
        self.buffer = deque(maxlen=self.window_size)
        print("-> Brain Ready.\n")

    def predict(self, full_sensor_state):
        # 1. FILTERN: Nur die relevanten Sensoren herauspicken
        # full_sensor_state hat 60 Werte, filtered_data hat z.B. nur 36
        filtered_data = full_sensor_state[self.input_indices]
        
        # 2. Skalieren
        input_raw = filtered_data.reshape(1, -1)
        input_scaled = self.scaler_x.transform(input_raw)
        
        # 3. Buffer & Predict
        self.buffer.append(input_scaled[0])
        if len(self.buffer) < self.window_size: return None
            
        input_tensor = torch.FloatTensor(np.array([self.buffer])).to(self.device)
        with torch.no_grad():
            out = self.model(input_tensor).cpu().numpy()
            
        # 4. Output
        if self.use_scaler_y:
            return self.scaler_y.inverse_transform(out)[0]
        else:
            return out[0] * self.angle_max

# ================= SERIAL READER (STANDARD) =================
class SerialReader:
    def __init__(self, port, baud, shared_state_array):
        self.port = port
        self.baud = baud
        self.running = False
        self.thread = None
        self.state = shared_state_array 
        self.HEADER = b"\xAA\x55"
        self.TXYZ_SIZE = 16 
        self.PAYLOAD_SIZE = NUM_SENSORS_PER_BOARD * self.TXYZ_SIZE
        self.FEATURES_PER_SENSOR = 4 

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        print(f"Öffne Serial {self.port}...")
        ser = None
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.1)
            ser.reset_input_buffer()
            while self.running:
                if ser.read(1) != self.HEADER[:1]: continue
                if ser.read(1) != self.HEADER[1:]: continue
                bid = ser.read(1)
                if not bid: continue
                board_id = bid[0]
                payload = ser.read(self.PAYLOAD_SIZE)
                if len(payload) != self.PAYLOAD_SIZE: continue
                if board_id >= NUM_BOARDS: continue 
                
                # Interleaved schreiben
                board_start = board_id * NUM_SENSORS_PER_BOARD * self.FEATURES_PER_SENSOR
                for s in range(NUM_SENSORS_PER_BOARD):
                    off = s * self.TXYZ_SIZE
                    _, x, y, z = struct.unpack_from("<ffff", payload, off)
                    mag = math.sqrt(x*x + y*y + z*z)
                    
                    idx = board_start + (s * 4)
                    self.state[idx] = x
                    self.state[idx+1] = y
                    self.state[idx+2] = z
                    self.state[idx+3] = mag
        except Exception as e:
            print(f"Serial Error: {e}")
            self.running = False
        finally:
            if ser: ser.close()

# ================= MAIN =================
def main():
    # Array immer für ALLE Sensoren anlegen (Reader füllt alles)
    total_inputs = NUM_BOARDS * NUM_SENSORS_PER_BOARD * NUM_FEATURES_PER_SENSOR
    global_state = np.zeros(total_inputs, dtype=np.float32)
    
    try:
        brain = NeuralBrain(RUN_FOLDER_NAME)
    except Exception as e:
        print(e)
        return

    reader = SerialReader(SERIAL_PORT, BAUD_RATE, global_state)
    reader.start()
    
    print("--- LIVE ---")
    try:
        while reader.running:
            t_s = time.perf_counter()
            # Kopie ziehen
            data = global_state.copy()
            if np.all(data == 0): 
                time.sleep(0.1); continue
            
            # Brain filtert sich das raus, was es braucht
            angles = brain.predict(data)
            
            if angles is not None:
                print(f"\rWinkel: {[f'{a:6.1f}' for a in angles]}  ({(time.perf_counter()-t_s)*1000:.1f}ms)", end="")
            else:
                print("\rPuffern...", end="")
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        reader.running = False

if __name__ == "__main__":
    main()