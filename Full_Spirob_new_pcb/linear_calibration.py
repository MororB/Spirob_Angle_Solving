"""
linear_calibration.py
=====================
Winkelschätzung via Linearisierung.

Für jeden Winkel (theta_1 .. theta_13) wird folgende lineare Abbildung
aus den Trainingsdaten bestimmt anhand von festen +30° und -30° Referenzpunkten:

    angle = slope * mag_value + intercept

Das Ergebnis wird als 'linear_calib.json' gespeichert.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ===================== CONFIGURATION =====================
# ── Run-Ordner ──────────────────────────────────────────────────────────────
# Trage hier alle Run-Ordner ein, die für die Kalibrierung genutzt werden sollen.
RUN_DIRS = [
    r"runs\2026-02-19_10-14-37",   # <-- Runs hier eintragen
    r"runs\2026-02-16_16-58-18",
    r"runs\2026-02-16_15-19-31",
    #r"runs\2026-02-16_12-20-17",
]

# ── Achsen ──────────────────────────────────────────────────────────────────
MAG_AXES = ["magX", "magY", "magZ"]

# ── Ziel-Winkel ─────────────────────────────────────────────────────────────
THETA_COLS   = [f"theta_{i}" for i in range(1, 14)]   # theta_1 .. theta_13
CALIB_OUTPUT = "linear_calib.json"

# ── Sensor → Theta Zuordnung (physischer Aufbau, siehe Spirob_Aufbau.md) ───
# Jeder Sensor misst NUR die Gelenke direkt neben ihm (durch Magnet).
# Sensor-IDs = I²C XOR-IDs wie sie im Datenstrom erscheinen.
#
#   Aufbau:  Sensor → Theta → Magnet → Theta → Sensor → ...
#   Sensor im Segment N misst:
#     - Theta darunter (Magnet unterhalb, falls vorhanden)
#     - Theta darüber  (Magnet oberhalb)
#
# theta_12 und theta_13 können nicht gemessen werden (keine Sensoren dort).
SENSOR_THETA_MAP = {
    0: ["theta_1"],                    # Seg 1 (unterstes), nur Magnet oben
    1: ["theta_2", "theta_3"],         # Seg 3, Magnet unten UND oben
    2: ["theta_4", "theta_5"],         # Seg 5
    3: ["theta_6", "theta_7"],         # Seg 7
    4: ["theta_8", "theta_9"],         # Seg 9
    5: ["theta_10", "theta_11"],       # Seg 11 (letzter Sensor)
}

# Automatisch abgeleitete Listen für Kompatibilität
SENSOR_IDS = sorted(SENSOR_THETA_MAP.keys())
MEASURABLE_THETAS = sorted(
    {t for thetas in SENSOR_THETA_MAP.values() for t in thetas},
    key=lambda x: int(x.split("_")[1])
)

# ── Fixpunkte für Kalibrierung ──────────────────────────────────────────────
TARGET_MIN_ANGLE = -30.0  # Erster Fixpunkt (z.B. -30 Grad)
TARGET_MAX_ANGLE =  30.0  # Zweiter Fixpunkt (z.B. +30 Grad)
ANGLE_TOLERANCE  =   5.0  # Toleranz (z.B. +/- 5 Grad um den Fixpunkt suchen)

# Nur Kalibrierpaare mit R² über diesem Wert speichern (negative R² = unsinnig)
R2_MIN_SAVE = 0.0


# ===================== HELPER =====================

def load_merged_csv(run_dir: Path) -> pd.DataFrame:
    """Lädt merged.csv, oder baut es aus sensors.csv + labels.csv zusammen."""
    merged_path = run_dir / "merged_smoothed.csv"
    if merged_path.exists():
        print(f"  Lade merged.csv ... ", end="")
        df = pd.read_csv(merged_path)
        print(f"{len(df)} Zeilen")
        return df

    sensors_path = run_dir / "sensors.csv"
    labels_path  = run_dir / "labels.csv"
    if not sensors_path.exists():
        raise FileNotFoundError(f"Keine sensors.csv in {run_dir}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Keine labels.csv in {run_dir}")

    print("  Kein merged.csv gefunden – lade sensors.csv + labels.csv ...")
    sensors = pd.read_csv(sensors_path)
    labels  = pd.read_csv(labels_path)

    labels = labels.rename(columns={"t_estimated_ns": "t_label_ns"})
    labels = labels.sort_values("t_label_ns").reset_index(drop=True)
    sensors = sensors.sort_values("t_pc_ns").reset_index(drop=True)

    merged = pd.merge_asof(
        sensors,
        labels[["t_label_ns"] + [c for c in THETA_COLS if c in labels.columns]],
        left_on="t_pc_ns",
        right_on="t_label_ns",
        direction="nearest"
    )
    print(f"  Gejoined: {len(merged)} Zeilen")
    return merged


def load_all_runs(run_dirs: list) -> pd.DataFrame:
    """Lädt alle konfigurierten Run-Ordner und kombiniert sie zu einem DataFrame."""
    base = Path(__file__).parent
    frames = []
    for raw_path in run_dirs:
        run_dir = Path(raw_path)
        if not run_dir.is_absolute():
            run_dir = base / run_dir
        if not run_dir.exists():
            print(f"  [SKIP] Ordner nicht gefunden: {run_dir}")
            continue
        print(f"\n  ┌─ Lade: {run_dir.name}")
        try:
            df = load_merged_csv(run_dir)
            df["_run"] = run_dir.name
            frames.append(df)
            print(f"  └─ OK: {len(df)} Zeilen")
        except Exception as e:
            print(f"  └─ FEHLER: {e}")

    if not frames:
        raise RuntimeError("Kein einziger Run konnte geladen werden!")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Gesamt: {len(combined)} Zeilen aus {len(frames)} Run(s)")
    return combined


# ===================== CORE CALIBRATION =====================

def calibrate(df: pd.DataFrame, axes: list) -> dict:
    """
    Kalibriert NUR physikalisch sinnvolle Sensor→Theta-Paare
    (definiert in SENSOR_THETA_MAP, siehe Spirob_Aufbau.md).

    Für jede Kombination wird die Linearisierung aus Fixpunkten
    bei ±30° berechnet.
    """
    mappings = []

    available_theta = [c for c in THETA_COLS if c in df.columns]
    if not available_theta:
        raise RuntimeError("Keine theta_*-Spalten im Datensatz!")

    # Nur physikalisch messbare Thetas kalibrieren
    calibratable = [t for t in MEASURABLE_THETAS if t in available_theta]
    unmeasurable = [t for t in THETA_COLS if t not in MEASURABLE_THETAS]

    n_pairs = sum(
        len(axes) for thetas in SENSOR_THETA_MAP.values()
        for t in thetas if t in available_theta
    )
    print(f"\nKalibriere {len(SENSOR_THETA_MAP)} Sensoren × {len(axes)} Achsen")
    print(f"  Messbare Thetas: {calibratable}")
    if unmeasurable:
        print(f"  ⚠️  Nicht messbar (keine Sensoren): {unmeasurable}")
    print(f"  Erwartete Paare: {n_pairs}")
    print(f"  Fixpunkte: {TARGET_MIN_ANGLE}° und {TARGET_MAX_ANGLE}° (Toleranz ±{ANGLE_TOLERANCE}°)\n")

    for sensor_id, theta_list in SENSOR_THETA_MAP.items():
        sensor_df = df[df["sensor_id"] == sensor_id].copy()
        if len(sensor_df) == 0:
            print(f"  ⚠️  Sensor {sensor_id}: keine Daten gefunden!")
            continue

        for theta_col in theta_list:
            if theta_col not in available_theta:
                print(f"  SKIP  sensor{sensor_id:2d} → {theta_col:8s}: nicht in Labels vorhanden")
                continue

            for axis in axes:
                if axis not in sensor_df.columns:
                    continue

                valid = sensor_df[[axis, theta_col]].dropna()
                if len(valid) == 0:
                    continue

                mag_vals   = valid[axis].values
                angle_vals = valid[theta_col].values

                # Daten filtern: Im Kalibrierbereich [-30°, +30°]
                mask_valid = (angle_vals >= TARGET_MIN_ANGLE - ANGLE_TOLERANCE) & (angle_vals <= TARGET_MAX_ANGLE + ANGLE_TOLERANCE)
                
                if np.sum(mask_valid) < 3:
                    print(f"  SKIP  sensor{sensor_id:2d} {axis:4s} → {theta_col:8s}: Zu wenig Datenpunkte im Range")
                    continue

                # Lineare Regression über ALLE Datenpunkte im Bereich
                mag_valid = mag_vals[mask_valid]
                angle_valid = angle_vals[mask_valid]
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(mag_valid, angle_valid)
                r2 = r_value ** 2

                # Zu Referenzzwecken: die Extrempunkte bestimmen
                mag_at_min = float(np.min(mag_valid))
                mag_at_max = float(np.max(mag_valid))
                real_angle_min = float(np.mean(angle_valid[mag_valid == mag_at_min]))
                real_angle_max = float(np.mean(angle_valid[mag_valid == mag_at_max]))

                status = "✓" if r2 > 0.3 else "~" if r2 > 0 else "✗"
                print(f"  {status}  sensor{sensor_id:2d} {axis:4s} → {theta_col:8s}: "
                      f"R²={r2:+.4f}  slope={slope:.4f}")

                # Nur Mappings mit ausreichend gutem R² speichern
                if r2 <= R2_MIN_SAVE:
                    continue

                mappings.append({
                    "sensor_id":    sensor_id,
                    "axis":         axis,
                    "target_angle": theta_col,
                    "mag_min":      round(mag_at_min, 4),
                    "mag_max":      round(mag_at_max, 4),
                    "angle_at_min": round(real_angle_min, 4),
                    "angle_at_max": round(real_angle_max, 4),
                    "slope":        round(slope, 6),
                    "intercept":    round(intercept, 6),
                    "r2":           round(r2, 4)
                })

    print(f"\n  → {len(mappings)} Kalibrierpaare gespeichert (R² > {R2_MIN_SAVE})")
    return {"mappings": mappings}


# ===================== MAIN =====================

def main():
    print("=" * 60)
    print("   Linearisierung Winkelkalibrierung (±30° Methode)")
    print("=" * 60)
    print(f"\nKonfigurierte Runs ({len(RUN_DIRS)}):")
    for p in RUN_DIRS:
        print(f"  • {p}")
    print(f"Achsen: {MAG_AXES}")
    print(f"Output: {CALIB_OUTPUT}")

    print(f"\nSensor → Theta Zuordnung (siehe Spirob_Aufbau.md):")
    for sid, thetas in SENSOR_THETA_MAP.items():
        print(f"  Sensor {sid} → {', '.join(thetas)}")
    print(f"  ⚠️  Nicht messbar: theta_12, theta_13")
    print("-" * 60)

    # 1. Alle Runs laden und kombinieren
    df = load_all_runs(RUN_DIRS)

    # 2. Sensor-IDs in den Daten prüfen
    data_sensor_ids = sorted(df["sensor_id"].unique())
    print(f"\n  Sensor-IDs in Daten: {data_sensor_ids}")
    missing = [s for s in SENSOR_IDS if s not in data_sensor_ids]
    if missing:
        print(f"  ⚠️  Fehlende Sensoren: {missing} (in SENSOR_THETA_MAP definiert, aber nicht in Daten!)")

    # 3. Kalibrierung berechnen
    calib = calibrate(df, MAG_AXES)

    # 4. Speichern
    out_path = Path(__file__).parent / CALIB_OUTPUT
    with open(out_path, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\n✅ Kalibrierung gespeichert: {out_path}")

    # 5. Zusammenfassung
    mappings = calib["mappings"]
    if mappings:
        r2_vals = [m["r2"] for m in mappings]
        good = [m for m in mappings if m["r2"] > 0.3]
        ok   = [m for m in mappings if 0 < m["r2"] <= 0.3]
        bad  = [m for m in mappings if m["r2"] <= 0]

        print(f"\n📊 Zusammenfassung ({len(mappings)} Kalibrierpaare):")
        print(f"   ✓ Gut (R² > 0.3):  {len(good)}")
        print(f"   ~ OK  (0 < R² ≤ 0.3): {len(ok)}")
        print(f"   ✗ Schlecht (R² ≤ 0):  {len(bad)}")
        print(f"   R² Median: {np.median(r2_vals):.3f}  |  Max: {np.max(r2_vals):.3f}")

        if good:
            print(f"\n   Beste Paare:")
            for m in sorted(good, key=lambda x: -x["r2"])[:5]:
                print(f"     sensor{m['sensor_id']} {m['axis']} → {m['target_angle']}: R²={m['r2']:.4f}")
    else:
        print("\n⚠️  Keine Kalibrierpaare erstellt!")

    print("\nFertig!")


if __name__ == "__main__":
    main()
