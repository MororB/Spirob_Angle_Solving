"""
linear_live_test.py
===================
Live-Winkelschätzung via Linearisierung.

Lädt 'linear_calib.json' (erstellt von linear_calibration.py)
und berechnet die Gelenkwinkel in Echtzeit aus den Sensordaten:

    angle = slope * mag_value + intercept

Unterstützt Console- und PyQtGraph-Ausgabe.
"""

import serial
import struct
import threading
import time
import json
import os
import numpy as np
from pathlib import Path
from collections import deque

# PyQtGraph for Live-Visualization (optional)
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtCore
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

# ===================== CONFIGURATION =====================
SERIAL_PORT   = "COM6"
BAUD_RATE     = 1_000_000
CALIB_FILE    = "linear_calib.json"
MAX_BUFFER_FRAMES = 20

# Mittelungs-Strategie wenn mehrere Sensoren/Achsen denselben Winkel liefern
# "mean"   = gewichteter Mittelwert (r² als Gewicht)
# "best"   = nur beste r²
MERGE_STRATEGY = "mean"

# Smoothing
SMOOTH_WINDOW = 5   # 1 = kein Smoothing

# Binary Frame Constants (identisch zu live_test.py)
FRAME_HEADER      = bytes([0xAA, 0x55])
FRAME_HEADER_SIZE = 2
FRAME_META_SIZE   = 9   # frame_id(4) + t_esp_us(4) + num_sensors(1)
SENSOR_PACKET_SIZE = 25  # sensor_id(1) + 6×float32(24)
FIELD_NAMES = ["accX", "accY", "accZ", "magX", "magY", "magZ"]


# ===================== CALIBRATION LOADING =====================

class LinearCalib:
    """Hält alle Kalibrierpaare und berechnet Winkel aus Sensor-Rohwerten."""

    # Nur Mappings mit R² über diesem Schwellwert verwenden
    R2_THRESHOLD = 0.0
    # Physikalisches Limit der Gelenke – Vorhersagen darüber werden verworfen
    MAX_ANGLE = 60.0

    def __init__(self, calib_path: str):
        with open(calib_path, "r") as f:
            data = json.load(f)

        all_mappings = data["mappings"]

        # Nur Mappings mit positivem R² behalten (Rest ist physikalisch unsinnig)
        self.mappings = [m for m in all_mappings if m.get("r2", 0) > self.R2_THRESHOLD]
        skipped = len(all_mappings) - len(self.mappings)

        # Indexierung: angle_col → Liste von Mappings (für Multi-Sensor-Fusion)
        self.by_angle: dict[str, list] = {}
        for m in self.mappings:
            key = m["target_angle"]
            self.by_angle.setdefault(key, []).append(m)

        self.angle_cols = sorted(self.by_angle.keys(),
                                 key=lambda x: int(x.split("_")[1]))
        print(f"✅ Kalibrierung geladen: {len(self.mappings)} Paare aktiv, "
              f"{skipped} verworfen (R² ≤ {self.R2_THRESHOLD}), "
              f"{len(self.angle_cols)} Winkel")
        for col in self.angle_cols:
            maps = self.by_angle[col]
            best_r2 = max(m["r2"] for m in maps)
            sensors = set(m["sensor_id"] for m in maps)
            print(f"    {col}: {len(maps)} Paare, Sensoren {sensors}, bestes R²={best_r2:.4f}")

    def predict(self, sensor_values: dict[int, dict[str, float]],
                strategy: str = "mean") -> dict[str, float]:
        """
        sensor_values: {sensor_id: {"magX": ..., "magY": ..., "magZ": ..., ...}}
        Gibt {theta_col: angle_degrees} zurück.
        """
        results = {}
        for theta_col, maps in self.by_angle.items():
            estimates = []
            weights   = []
            for m in maps:
                sid  = m["sensor_id"]
                axis = m["axis"]
                if sid not in sensor_values:
                    continue
                val = sensor_values[sid].get(axis)
                if val is None:
                    continue

                pred = m["slope"] * val + m["intercept"]

                # Vorhersagen außerhalb des physikalischen Bereichs verwerfen
                if abs(pred) > self.MAX_ANGLE:
                    continue

                w = m.get("r2", 0.01)   # R² als Gewicht (bereits > 0 durch Filter)
                estimates.append(pred)
                weights.append(w)

            if not estimates:
                results[theta_col] = float("nan")
            elif strategy == "best":
                best_idx = int(np.argmax(weights))
                results[theta_col] = estimates[best_idx]
            else:  # "mean"
                w = np.array(weights)
                results[theta_col] = float(np.average(estimates, weights=w))

        return results



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
            self.buffer  = self.buffer[frame_size:]

            sensors = {}
            for i in range(num_sensors):
                offset = 11 + i * SENSOR_PACKET_SIZE
                sid  = frame_bytes[offset]
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
        self.frame_queue: deque = deque(maxlen=MAX_BUFFER_FRAMES)
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
                else:
                    time.sleep(0.001)
        except Exception as e:
            print(f"\n[Serial Error] {e}")
            self.running = False


# ===================== OFFLINE REPLAY (CSV-Test) =====================

def replay_from_csv(calib: LinearCalib, csv_path: str,
                    strategy: str = "mean", smooth_window: int = 1, log_file=None):
    """
    Simuliert den Live-Test mit einer vorhandenen sensors.csv.
    Nützlich zum Testen ohne Hardware.
    """
    import pandas as pd

    print(f"\n[Offline-Replay] Lade {csv_path} ...")
    df = pd.read_csv(csv_path)

    smooth_bufs = {col: deque(maxlen=smooth_window) for col in calib.angle_cols}

    frame_groups = df.groupby("frame_id")
    print(f"  {len(frame_groups)} Frames geladen. Strg+C zum Stoppen.\n")

    try:
        for frame_id, group in frame_groups:
            sensor_values = {}
            for _, row in group.iterrows():
                sid = int(row["sensor_id"])
                sensor_values[sid] = {
                    "magX": row.get("magX", 0.0),
                    "magY": row.get("magY", 0.0),
                    "magZ": row.get("magZ", 0.0),
                    "accX": row.get("accX", 0.0),
                    "accY": row.get("accY", 0.0),
                    "accZ": row.get("accZ", 0.0),
                }

            angles = calib.predict(sensor_values, strategy)

            # Smoothing
            smoothed = {}
            for col, val in angles.items():
                if not np.isnan(val):
                    smooth_bufs[col].append(val)
                if smooth_bufs[col]:
                    smoothed[col] = np.mean(smooth_bufs[col])
                else:
                    smoothed[col] = float("nan")

            if log_file:
                # Assuming ~100Hz if we don't have exactly timestamps for offline test replay easily accessible here,
                # but we can use frame_id as fake time, or actual time.time()
                row = [f"{time.time():.6f}"]
                for col in calib.angle_cols:
                    row.append(f"{smoothed.get(col, float('nan')):.4f}")
                log_file.write("," .join(row) + "\n")

            angle_str = "  ".join(
                f"{c}: {v:+6.1f}°" if not np.isnan(v) else f"{c}:    NaN "
                for c, v in smoothed.items()
            )
            print(f"\rFrame {frame_id:5d}  {angle_str}", end="", flush=True)
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n\n[Replay gestoppt]")


# ===================== MAIN =====================

def main():
    print("=" * 60)
    print("   Linear Live-Test — Winkelschätzung via Linearisierung")
    print("=" * 60)

    # Kalibrierungsdatei suchen
    calib_search = [
        Path(CALIB_FILE),
        Path(__file__).parent / CALIB_FILE,
    ]
    calib_path = None
    for p in calib_search:
        if p.exists():
            calib_path = str(p)
            break
    if calib_path is None:
        alt = input(f"\nKeine {CALIB_FILE} gefunden. Pfad eingeben: ").strip()
        if not Path(alt).exists():
            print("Datei nicht gefunden."); return
        calib_path = alt

    calib = LinearCalib(calib_path)
    print(f"  Winkel: {calib.angle_cols}")

    # Modus wählen
    print("\nModus:")
    print("  [1] Live (Seriell)")
    print("  [2] Offline-Test (CSV-Datei)")
    mode_choice = input("Wahl [1]: ").strip()

    # Smoothing
    smooth_in = input(f"Smoothing-Fenster [Enter={SMOOTH_WINDOW}]: ").strip()
    smooth_window = int(smooth_in) if smooth_in.isdigit() else SMOOTH_WINDOW

    # Strategie
    print(f"\nFusions-Strategie:")
    print(f"  [m] Gewichteter Mittelwert (r²-Gewicht)")
    print(f"  [b] Nur bestes Mapping (max. r²)")
    strat_in = input("Strategie [Enter=m]: ").strip().lower()
    strategy = "best" if strat_in == "b" else "mean"

    log_in = input("\nWinkel-Vorhersagen in CSV loggen? [j/N]: ").strip().lower()
    do_log = (log_in == "j" or log_in == "y")
    log_file = None
    if do_log:
        log_dir = Path("Live_angles")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_filename = log_dir / f"live_angles_{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        log_file = open(log_filename, "w", encoding="utf-8")
        header = ["t_sys_s"] + calib.angle_cols
        log_file.write("," .join(header) + "\n")
        print(f"  [Log] Speichere in {log_filename}")

    # ── Offline-Replay ──────────────────────────────────────────
    if mode_choice == "2":
        csv_in = input("Pfad zur sensors.csv: ").strip()
        if not csv_in:
            # Letzten Run automatisch wählen
            runs = sorted(Path("runs").iterdir()) if Path("runs").exists() else []
            if runs:
                csv_in = str(runs[-1] / "sensors.csv")
                print(f"  → {csv_in}")
            else:
                print("Kein Run gefunden."); return
        replay_from_csv(calib, csv_in, strategy, smooth_window, log_file)
        if log_file:
            log_file.close()
        return

    # ── Live Serial ──────────────────────────────────────────────
    viz_in = input("\nVisualisierung: [c]onsole, [g]raph [Enter=c]: ").strip().lower()
    use_graph = (viz_in == "g" and HAS_PYQTGRAPH)
    if viz_in == "g" and not HAS_PYQTGRAPH:
        print("  [Warnung] PyQtGraph nicht verfügbar, verwende Console.")

    port_in = input(f"Serieller Port [Enter={SERIAL_PORT}]: ").strip()
    port = port_in if port_in else SERIAL_PORT

    smooth_bufs = {col: deque(maxlen=smooth_window) for col in calib.angle_cols}

    # PyQtGraph Setup
    app, curves, history, plot_widget = None, [], None, None
    if use_graph:
        app = QtWidgets.QApplication([])
        win  = pg.GraphicsLayoutWidget(title="Linear Live Angles")
        plot = win.addPlot(title="Gelenkwinkel")
        plot.setYRange(-60, 60)
        plot.setLabel("left", "Winkel [°]")
        plot.addLegend()
        history = deque(maxlen=300)
        colors = [(255, 80, 80), (80, 200, 80), (80, 120, 255),
                  (255, 200, 50), (200, 80, 255), (80, 220, 220),
                  (255, 140, 30), (180, 255, 80), (255, 80, 200),
                  (120, 200, 255), (255, 120, 120), (80, 255, 160),
                  (200, 160, 80)]
        for i, col in enumerate(calib.angle_cols):
            c = colors[i % len(colors)]
            curves.append(plot.plot(name=col, pen=pg.mkPen(c, width=2)))
        win.show()
        plot_widget = win

    reader = SerialReader(port, BAUD_RATE)
    reader.start()
    print(f"\n🚀 Läuft auf {port} | Strategie={strategy} | "
          f"Smoothing={smooth_window} | Strg+C zum Stoppen\n")

    try:
        while reader.is_alive():
            if use_graph:
                app.processEvents()

            while reader.frame_queue:
                frame = reader.frame_queue.popleft()
                sensor_values = frame["sensors"]

                angles = calib.predict(sensor_values, strategy)

                # Smoothing
                smoothed = {}
                for col, val in angles.items():
                    if not np.isnan(val):
                        smooth_bufs[col].append(val)
                    if smooth_bufs[col]:
                        smoothed[col] = np.mean(smooth_bufs[col])
                    else:
                        smoothed[col] = float("nan")

                if log_file:
                    row = [f"{time.time():.6f}"]
                    for col in calib.angle_cols:
                        row.append(f"{smoothed.get(col, float('nan')):.4f}")
                    log_file.write("," .join(row) + "\n")

                if use_graph:
                    history.append([smoothed.get(c, float("nan"))
                                     for c in calib.angle_cols])
                    arr = np.array(list(history))
                    for i, curve in enumerate(curves):
                        col_data = arr[:, i]
                        valid = ~np.isnan(col_data)
                        if np.any(valid):
                            curve.setData(col_data)
                else:
                    parts = [
                        f"{c.split('_')[1]:>2}: {v:+6.1f}°"
                        if not np.isnan(v) else f"{c.split('_')[1]:>2}:   NaN "
                        for c, v in smoothed.items()
                    ]
                    print(f"\r{'  '.join(parts)}", end="", flush=True)

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n\n[Gestoppt]")
    finally:
        reader.running = False
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    main()
