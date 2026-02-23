# verify_run.py
# Prüft einen Run-Ordner (CSV-only) auf Plausibilität:
# - Dateien vorhanden?
# - Spaltentypen OK (theta numeric)?
# - Anteil quality==1
# - Zeitfenster-Overlap Sensor <-> Labels
# - grobe Sensorrate + Videofps (aus timestamps gemessen)

import argparse
from pathlib import Path
import polars as pl

THETA_COLS = ["theta_1", "theta_2", "theta_3", "theta_4"]

def human_ns(ns: int) -> str:
    # ns -> ms (nur fürs Gefühl)
    return f"{ns/1e6:.2f} ms"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help=r"Run-Ordner, z.B. .\runs\2025-12-08_13-41-29\ ")
    args = ap.parse_args()
    run_dir = Path(args.run)

    sensors_path = run_dir / "sensors.csv"
    labels_path  = run_dir / "labels.csv"
    frames_path  = run_dir / "frames.csv"
    video_path   = run_dir / "video.mp4"
    meta_path    = run_dir / "meta.yaml"

    print(f"\n=== VERIFY RUN: {run_dir} ===")
    for p in [meta_path, video_path, sensors_path, labels_path, frames_path]:
        print(f"{'OK ' if p.exists() else 'MISSING'}  {p.name}")

    if not sensors_path.exists() or not labels_path.exists() or not frames_path.exists():
        print("\nAbbruch: sensors.csv/labels.csv/frames.csv fehlt.")
        return

    # ---------- Load sensors ----------
    s = pl.read_csv(sensors_path, infer_schema_length=5000)

    # Casts (robust gegen Strings)
    s = s.with_columns([
        pl.col("t_ns").cast(pl.Int64, strict=False),
        pl.col("board_id").cast(pl.Int64, strict=False),
        pl.col("sensor_id").cast(pl.Int64, strict=False),
        pl.col("t").cast(pl.Float64, strict=False),
        pl.col("x").cast(pl.Float64, strict=False),
        pl.col("y").cast(pl.Float64, strict=False),
        pl.col("z").cast(pl.Float64, strict=False),
    ]).drop_nulls(["t_ns"]).sort("t_ns")

    # ---------- Load labels ----------
    l = pl.read_csv(labels_path, infer_schema_length=5000)

    # quality fallback
    if "quality" not in l.columns:
        l = l.with_columns(pl.lit(1).alias("quality"))

    l = l.with_columns([
        pl.col("t_frame_ns").cast(pl.Int64, strict=False),
        pl.col("quality").cast(pl.Int64, strict=False),
        *[pl.col(c).cast(pl.Float64, strict=False) for c in THETA_COLS if c in l.columns],
    ]).drop_nulls(["t_frame_ns"]).sort("t_frame_ns")

    # ---------- Load frames ----------
    f = pl.read_csv(frames_path, infer_schema_length=5000)
    f = f.with_columns([
        pl.col("frame_idx").cast(pl.Int64, strict=False),
        pl.col("t_frame_ns").cast(pl.Int64, strict=False),
    ]).drop_nulls(["t_frame_ns"]).sort("t_frame_ns")

    # ---------- Basic counts ----------
    print("\n--- Counts ---")
    print("Sensors rows:", s.height)
    print("Labels rows :", l.height)
    print("Frames rows :", f.height)

    # ---------- Labels quality stats ----------
    print("\n--- Label quality ---")
    q_counts = l.group_by("quality").len().sort("quality")
    print(q_counts)

    good = l.filter(pl.col("quality") == 1)
    print("Good label rows (quality==1):", good.height)

    # ---------- Theta numeric sanity ----------
    print("\n--- Theta sanity ---")
    for c in THETA_COLS:
        if c not in l.columns:
            print(f"MISSING column: {c}")
            continue
        total = l.height
        finite = l.filter(pl.col(c).is_not_null() & pl.col(c).is_finite()).height
        nulls = l.select(pl.col(c).is_null().sum()).item()
        print(f"{c}: finite={finite}/{total}  nulls={nulls}")

    # ---------- Time windows & overlap ----------
    print("\n--- Time windows ---")
    s_min = s.select(pl.col("t_ns").min()).item()
    s_max = s.select(pl.col("t_ns").max()).item()
    l_min = l.select(pl.col("t_frame_ns").min()).item()
    l_max = l.select(pl.col("t_frame_ns").max()).item()

    print(f"Sensors: {human_ns(s_min)} .. {human_ns(s_max)}  (span {human_ns(s_max - s_min)})")
    print(f"Labels : {human_ns(l_min)} .. {human_ns(l_max)}  (span {human_ns(l_max - l_min)})")

    ov_min = max(s_min, l_min)
    ov_max = min(s_max, l_max)
    if ov_max > ov_min:
        print(f"Overlap: {human_ns(ov_min)} .. {human_ns(ov_max)}  (span {human_ns(ov_max - ov_min)})")
    else:
        print("Overlap: NONE  (Zeitstempel passen nicht zusammen?)")

    # ---------- Rough rates ----------
    print("\n--- Rough rates (measured) ---")
    # Sensor: n samples / seconds in window
    span_s = (s_max - s_min) / 1e9
    if span_s > 0:
        print(f"Sensor rows/sec (all sensors): {s.height / span_s:.1f} rows/s")
        # pro sensor-id+board
        per_chan = (
            s.group_by(["board_id","sensor_id"])
             .agg([pl.len().alias("n"),
                   (pl.col("t_ns").max() - pl.col("t_ns").min()).alias("span_ns")])
             .with_columns((pl.col("n") / (pl.col("span_ns")/1e9)).alias("rows_per_s"))
             .sort(["board_id","sensor_id"])
        )
        print("\nPer board/sensor approx rate:")
        print(per_chan.select(["board_id","sensor_id","rows_per_s"]).head(15))

    # Video fps measured from frames.csv
    f_min = f.select(pl.col("t_frame_ns").min()).item()
    f_max = f.select(pl.col("t_frame_ns").max()).item()
    span_f = (f_max - f_min) / 1e9
    if span_f > 0 and f.height > 1:
        fps = (f.height - 1) / span_f
        print(f"\nVideo measured FPS: {fps:.2f}")

    # ---------- Quick warnings ----------
    print("\n--- Warnings / Hints ---")
    if good.height < 2:
        print("! Sehr wenige quality==1 Frames. Prüfe: Marker sichtbar? quality wird gesetzt?")
    if ov_max <= ov_min:
        print("! Kein Overlap zwischen Sensor- und Label-Zeitfenster. Nutzen beide perf_counter_ns()?")
    for c in THETA_COLS:
        if c in l.columns:
            finite = l.filter(pl.col(c).is_not_null() & pl.col(c).is_finite()).height
            if finite < l.height * 0.2:
                print(f"! {c} hat viele ungültige Werte -> Marker oft nicht komplett sichtbar.")

    print("\nDone.\n")

if __name__ == "__main__":
    main()
