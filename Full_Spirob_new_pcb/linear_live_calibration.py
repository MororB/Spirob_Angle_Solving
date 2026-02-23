"""
linear_live_calibration.py
==========================
Interaktive Live-Kalibrierung pro Gelenk.

Anleitung:
1. Script starten
2. Für jedes Gelenk (1-11):
   - Fahre auf -30°, drücke ENTER (sammelt ~200 Frames)
   - Fahre auf +30°, drücke ENTER (sammelt ~200 Frames)
3. Script berechnet lineare Kalibration und speichert in linear_calib.json

Nutzt scipy.linregress für robuste Regression.
"""

import serial
import struct
import threading
import time
import json
import numpy as np
from pathlib import Path
from collections import deque
from scipy import stats

# ===================== CONFIGURATION =====================
SERIAL_PORT   = "COM6"
BAUD_RATE     = 1_000_000
CALIB_OUTPUT  = "linear_calib.json"
FRAMES_PER_POSITION = 200  # Wie viele Frames pro Extrempunkt sammeln

# Binary Frame Constants (identisch zu live_test.py)
FRAME_HEADER      = bytes([0xAA, 0x55])
FRAME_HEADER_SIZE = 2
FRAME_META_SIZE   = 9
SENSOR_PACKET_SIZE = 25
FIELD_NAMES = ["accX", "accY", "accZ", "magX", "magY", "magZ"]

# Sensor → Theta Zuordnung (aus linear_calibration.py)
SENSOR_THETA_MAP = {
    0: ["theta_1"],
    1: ["theta_2", "theta_3"],
    2: ["theta_4", "theta_5"],
    3: ["theta_6", "theta_7"],
    4: ["theta_8", "theta_9"],
    5: ["theta_10", "theta_11"],
}

# Physikalische Limits
TARGET_MIN_ANGLE = -30.0
TARGET_MAX_ANGLE = 30.0


# ===================== BINARY PARSER =====================

class BinaryFrameParser:
    def __init__(self, buffer_size: int = 8192):
        self.buffer = bytearray()
        self.buffer_size = buffer_size

    def add_data(self, data: bytes):
        self.buffer.extend(data)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]

    def extract_frames(self):
        frames = []
        while len(self.buffer) >= FRAME_HEADER_SIZE + FRAME_META_SIZE:
            idx = self.buffer.find(FRAME_HEADER)
            if idx == -1:
                break
            if idx > 0:
                self.buffer = self.buffer[idx:]

            if len(self.buffer) < FRAME_HEADER_SIZE + FRAME_META_SIZE:
                break

            frame_id, t_esp_us, num_sensors = struct.unpack(
                "<IIB", self.buffer[2:11])
            frame_size = (FRAME_HEADER_SIZE + FRAME_META_SIZE
                          + num_sensors * SENSOR_PACKET_SIZE)

            if len(self.buffer) < frame_size:
                break

            frame_bytes = self.buffer[:frame_size]
            self.buffer = self.buffer[frame_size:]

            sensors = {}
            for i in range(num_sensors):
                offset = 11 + i * SENSOR_PACKET_SIZE
                sid = frame_bytes[offset]
                vals = struct.unpack("<6f", frame_bytes[offset + 1:offset + 25])
                sensors[sid] = dict(zip(FIELD_NAMES, vals))

            frames.append({"id": frame_id, "sensors": sensors})
        return frames


# ===================== SERIAL READER =====================

class SerialReader(threading.Thread):
    def __init__(self, port: str, baud_rate: int):
        super().__init__(daemon=True)
        self.port, self.baud_rate = port, baud_rate
        self.running = False
        self.frame_queue: deque = deque(maxlen=500)
        self.parser = BinaryFrameParser()

    def run(self):
        self.running = True
        try:
            ser = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            ser.reset_input_buffer()
            print(f"✅ Seriell verbunden: {self.port} @ {self.baud_rate} baud")
            while self.running:
                if ser.in_waiting:
                    self.parser.add_data(ser.read(ser.in_waiting))
                    for frame in self.parser.extract_frames():
                        self.frame_queue.append(frame)
                else:
                    time.sleep(0.001)
        except Exception as e:
            print(f"\n❌ Serieller Fehler: {e}")
            self.running = False

    def collect_frames(self, n_frames: int) -> list:
        """Sammelt n Frames aus der Queue."""
        collected = []
        timeout = time.time() + 10.0  # Max 10 Sekunden Timeout
        
        while len(collected) < n_frames:
            if time.time() > timeout:
                print(f"⚠️  Timeout! Nur {len(collected)}/{n_frames} Frames gesammelt.")
                break
            
            try:
                frame = self.frame_queue.popleft()
                collected.append(frame)
                # Fortschritt anzeigen
                if len(collected) % 50 == 0:
                    print(f"  {len(collected)}/{n_frames} ...", end="\r")
            except IndexError:
                time.sleep(0.01)
        
        print(f"  ✓ {len(collected)} Frames gesammelt        ")
        return collected


# ===================== CALIBRATION =====================

def frames_to_sensor_values(frames: list) -> dict[int, dict[str, list]]:
    """
    Konvertiert eine Liste von Frames in:
    {sensor_id: {axis: [values...]}}
    """
    result = {}
    for frame in frames:
        for sensor_id, fields in frame["sensors"].items():
            if sensor_id not in result:
                result[sensor_id] = {axis: [] for axis in FIELD_NAMES}
            for axis, val in fields.items():
                result[sensor_id][axis].append(val)
    return result


def calibrate_single_joint(sensor_id: int, theta_col: str, 
                          data_min: dict, data_max: dict) -> dict:
    """
    Kalibiert ein einzelnes Gelenk (Sensor + Theta) mit Mindest- und Maximalunterschiede.
    
    data_min: {axis: [values...]} bei TARGET_MIN_ANGLE (-30°)
    data_max: {axis: [values...]} bei TARGET_MAX_ANGLE (+30°)
    """
    mappings = []
    axes = ["magX", "magY", "magZ"]
    
    for axis in axes:
        if axis not in data_min or axis not in data_max:
            continue
        
        vals_min = np.array(data_min[axis])
        vals_max = np.array(data_max[axis])
        
        if len(vals_min) < 3 or len(vals_max) < 3:
            print(f"      ⚠️  {axis}: Zu wenig Datenpunkte")
            continue
        
        # Alle Werte kombinieren mit bekannten Labels
        all_mag_vals = np.concatenate([vals_min, vals_max])
        all_angles = np.concatenate([
            np.full(len(vals_min), TARGET_MIN_ANGLE),
            np.full(len(vals_max), TARGET_MAX_ANGLE)
        ])
        
        # Lineare Regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_mag_vals, all_angles)
        r2 = r_value ** 2
        
        status = "✓" if r2 > 0.3 else "~" if r2 > 0 else "✗"
        print(f"      {status} {axis:5s}: R²={r2:+.4f}  slope={slope:.6f}  intercept={intercept:.6f}")
        
        if r2 > 0:  # Nur positive R² speichern
            mappings.append({
                "sensor_id": sensor_id,
                "axis": axis,
                "target_angle": theta_col,
                "mag_min": round(float(np.min(all_mag_vals)), 4),
                "mag_max": round(float(np.max(all_mag_vals)), 4),
                "angle_at_min": TARGET_MIN_ANGLE,
                "angle_at_max": TARGET_MAX_ANGLE,
                "slope": round(slope, 6),
                "intercept": round(intercept, 6),
                "r2": round(r2, 4)
            })
    
    return mappings


# ===================== MAIN =====================

def main():
    print("=" * 70)
    print("   Live-Kalibrierung für Spirob-Gelenke (Linearisierung)")
    print("=" * 70)
    print(f"\nSerial: {SERIAL_PORT} @ {BAUD_RATE} baud")
    print(f"Output: {CALIB_OUTPUT}")
    print(f"Frames pro Position: {FRAMES_PER_POSITION}")
    print(f"Fixpunkte: {TARGET_MIN_ANGLE}° und {TARGET_MAX_ANGLE}°")
    print("-" * 70)

    # Serial-Verbindung starten
    reader = SerialReader(SERIAL_PORT, BAUD_RATE)
    reader.start()
    time.sleep(1)

    if not reader.running:
        print("\n❌ Konnte Seriellverbindung nicht öffnen!")
        return

    all_mappings = []
    
    # Für jede Sensor/Theta-Kombination kalibrieren
    for sensor_id, theta_list in SENSOR_THETA_MAP.items():
        for theta_col in theta_list:
            print(f"\n📍 Sensor {sensor_id} → {theta_col}")
            print(f"  ┌─ Fahre auf {TARGET_MIN_ANGLE}° ...", end=" ")
            input("(Drücke ENTER wenn bereit)")
            
            print(f"  │ Sammle Daten...")
            frames_min = reader.collect_frames(FRAMES_PER_POSITION)
            if not frames_min:
                print(f"  └─ ❌ Keine Frames gesammelt! Springe zu nächstem Gelenk.")
                continue
            
            data_min = frames_to_sensor_values(frames_min)
            if sensor_id not in data_min:
                print(f"  └─ ❌ Sensor {sensor_id} keine Daten! Springe zu nächstem Gelenk.")
                continue
            
            print(f"  ├─ Fahre auf {TARGET_MAX_ANGLE}° ...", end=" ")
            input("(Drücke ENTER wenn bereit)")
            
            print(f"  │ Sammle Daten...")
            frames_max = reader.collect_frames(FRAMES_PER_POSITION)
            if not frames_max:
                print(f"  └─ ❌ Keine Frames gesammelt!")
                continue
            
            data_max = frames_to_sensor_values(frames_max)
            if sensor_id not in data_max:
                print(f"  └─ ❌ Sensor {sensor_id} keine Daten!")
                continue
            
            # Kalibration berechnen
            print(f"  └─ Berechne Kalibrierung...")
            mappings = calibrate_single_joint(sensor_id, theta_col,
                                             data_min[sensor_id],
                                             data_max[sensor_id])
            all_mappings.extend(mappings)
    
    # Serial-Reader stoppen
    reader.running = False
    reader.join(timeout=1)
    
    # Speichern
    output = {"mappings": all_mappings}
    out_path = Path(__file__).parent / CALIB_OUTPUT
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    
    # Zusammenfassung
    print("\n" + "=" * 70)
    print(f"✅ Kalibrierung gespeichert: {out_path}")
    print(f"   {len(all_mappings)} Mappings mit R² > 0")
    
    if all_mappings:
        r2_vals = [m["r2"] for m in all_mappings]
        good = len([m for m in all_mappings if m["r2"] > 0.3])
        ok   = len([m for m in all_mappings if 0 < m["r2"] <= 0.3])
        
        print(f"\n   ✓ Gut (R² > 0.3):        {good}")
        print(f"   ~ OK  (0 < R² ≤ 0.3):   {ok}")
        print(f"   R² Median: {np.median(r2_vals):.4f}  |  Max: {np.max(r2_vals):.4f}")
    
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Abgebrochen durch Benutzer.")
