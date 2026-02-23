import time
import struct
import threading
import serial
import numpy as np
import joblib
import math
from collections import deque

# ================= KONFIGURATION =================
SERIAL_PORT = "COM5"
BAUD_RATE = 1_000_000

# HIER DEN ORDNER MIT DEM POLYNOMIAL MODELL EINTRAGEN
MODEL_FOLDER_NAME = "polynomial_degree_3"  # Wird automatisch erstellt beim Speichern

# Hardware Setup (Bleibt immer gleich, wir filtern erst im Brain)
NUM_BOARDS = 3
NUM_SENSORS_PER_BOARD = 5
NUM_FEATURES_PER_SENSOR = 4  # x, y, z, mag

# ================= POLYNOMIAL BRAIN (OHNE LSTM!) =================
class PolynomialBrain:
    """
    Lädt ein trainiertes Polynomial Regression Modell und macht Echtzeit-Inferenz.
    VIEL SCHNELLER als LSTM - keine Buffer nötig!
    """
    def __init__(self, model_folder_name):
        import os
        base_path = os.path.join("./polynomial_models", model_folder_name)
        
        print(f"--- Lade Polynomial Brain: {model_folder_name} ---")
        
        # 1. Modell, Polynomial-Transformer und Scaler laden
        model_path = os.path.join(base_path, "model.pkl")
        poly_path = os.path.join(base_path, "poly_features.pkl")
        scaler_path = os.path.join(base_path, "scaler.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modell nicht gefunden: {model_path}")
        
        self.model = joblib.load(model_path)
        self.poly = joblib.load(poly_path)
        self.scaler = joblib.load(scaler_path)
        
        print(f"-> Modell geladen: {model_path}")
        print(f"-> Polynomial Grad: {self.poly.degree}")
        print(f"-> Features: {self.poly.n_output_features_}")
        print(f"-> Input Shape: {self.model.coef_.shape}")
        
        # 2. Welche Sensoren werden verwendet?
        # INFO: Das sollte idealerweise in einer config.json gespeichert sein
        # Für jetzt: Wir nehmen an, dass [0] verwendet wird (wie im Training)
        self.active_sensors = [0]  # TODO: Aus config.json laden
        
        # 3. Input-Maske berechnen
        self.input_indices = []
        for b in range(NUM_BOARDS):
            for s in range(NUM_SENSORS_PER_BOARD):
                if s in self.active_sensors:
                    global_sensor_idx = b * NUM_SENSORS_PER_BOARD + s
                    start_idx = global_sensor_idx * NUM_FEATURES_PER_SENSOR
                    # x, y, z, mag
                    self.input_indices.extend([start_idx, start_idx+1, start_idx+2, start_idx+3])
        
        self.input_indices = np.array(self.input_indices, dtype=np.int32)
        
        print(f"-> Nutze Sensoren pro Board: {self.active_sensors}")
        print(f"-> Input Indices: {len(self.input_indices)} Features")
        print("-> Brain Ready.\n")

    def predict(self, full_sensor_state):
        """
        Macht direkte Vorhersage ohne Buffer!
        
        Args:
            full_sensor_state: np.array mit 60 Werten (alle Sensoren)
            
        Returns:
            np.array mit 4 Winkeln [theta_1, theta_2, theta_3, theta_4]
        """
        # 1. FILTERN: Nur relevante Sensoren
        filtered_data = full_sensor_state[self.input_indices]
        
        # 2. Polynomial Features erstellen
        x_poly = self.poly.transform(filtered_data.reshape(1, -1))
        
        # 3. Skalieren
        x_scaled = self.scaler.transform(x_poly)
        
        # 4. Prediction (nur eine Matrixmultiplikation!)
        angles = self.model.predict(x_scaled)[0]
        
        return angles

# ================= SERIAL READER (IDENTISCH ZU LSTM) =================
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
    # Array für ALLE Sensoren (Reader füllt alles)
    total_inputs = NUM_BOARDS * NUM_SENSORS_PER_BOARD * NUM_FEATURES_PER_SENSOR
    global_state = np.zeros(total_inputs, dtype=np.float32)
    
    try:
        brain = PolynomialBrain(MODEL_FOLDER_NAME)
    except Exception as e:
        print(f"FEHLER beim Laden des Modells: {e}")
        return

    reader = SerialReader(SERIAL_PORT, BAUD_RATE, global_state)
    reader.start()
    
    print("=" * 60)
    print("POLYNOMIAL REGRESSION LIVE TEST")
    print("=" * 60)
    print("Drücke CTRL+C zum Beenden\n")
    
    try:
        while reader.running:
            t_s = time.perf_counter()
            
            # Kopie der aktuellen Sensordaten
            data = global_state.copy()
            
            # Warten bis Daten vorhanden
            if np.all(data == 0): 
                time.sleep(0.01)
                continue
            
            # DIREKTE Vorhersage (kein Buffer nötig!)
            angles = brain.predict(data)
            
            # Inferenzzeit messen
            inference_ms = (time.perf_counter() - t_s) * 1000
            
            # Ausgabe
            angle_str = ", ".join([f"{a:6.2f}°" for a in angles])
            print(f"\rθ: [{angle_str}]  |  {inference_ms:.3f} ms", end="")
            
            time.sleep(0.02)  # 50 Hz Update-Rate
            
    except KeyboardInterrupt:
        print("\n\nBeendet durch Benutzer.")
    finally:
        reader.running = False
        print("Serial Reader gestoppt.")

if __name__ == "__main__":
    main()
