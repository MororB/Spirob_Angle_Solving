import argparse
from pathlib import Path
import polars as pl
import math

# Updated Theta Columns for the new robot (13 joints)
THETA_COLS = [f"theta_{i+1}" for i in range(13)]

def merge_run(run_dir: Path, out_csv_name: str = "merged.csv", drop_outside: bool = True) -> Path:
    # 1. Paths Definitions
    sensors_path = run_dir / "sensors.csv"
    labels_path  = run_dir / "labels.csv"
    out_path     = run_dir / out_csv_name

    # Check existence
    if not sensors_path.exists():
        raise FileNotFoundError(f"Missing sensors.csv in {run_dir}")
    if not labels_path.exists():
        # Fallback to processed_angles.csv if label.csv doesn't strictly vary
        alt_path = run_dir / "processed_angles.csv"
        if alt_path.exists():
            print(f"[Merge] Using {alt_path.name} as labels source.")
            labels_path = alt_path
        else:
            raise FileNotFoundError(f"Missing labels.csv in {run_dir}")

    print(f"[Merge] Processing {run_dir.name}...")

    # 2. Load Data (Lazy for structure, then collect)
    # Sensors: t_pc_ns, sensor_id, accX, accY, accZ, magX, magY, magZ
    # Note: Using infer_schema_length=0 to force string then manual cast is safest for dirty CSVs,
    # but setting reasonable length usually works.
    s = pl.scan_csv(sensors_path, infer_schema_length=5000)
    
    # Cast Sensor columns explicitly
    cast_cols = [
        pl.col("t_pc_ns").cast(pl.Int64, strict=False),
        pl.col("sensor_id").cast(pl.Int64, strict=False),
        pl.col("accX").cast(pl.Float64, strict=False),
        pl.col("accY").cast(pl.Float64, strict=False),
        pl.col("accZ").cast(pl.Float64, strict=False),
        pl.col("magX").cast(pl.Float64, strict=False),
        pl.col("magY").cast(pl.Float64, strict=False),
        pl.col("magZ").cast(pl.Float64, strict=False),
    ]
    
    s_temp = s.collect()
    s = pl.LazyFrame(s_temp).with_columns(cast_cols).sort("t_pc_ns")

    # Labels: frame_idx, t_estimated_ns, theta_1 ... theta_13
    l = pl.scan_csv(labels_path, infer_schema_length=5000)
    
    # Cast Label columns explicitly
    l = l.with_columns([
        # Rename t_estimated_ns to a standard time column matching sensors for easier join logic later
        pl.col("t_estimated_ns").cast(pl.Int64, strict=False).alias("t_label_ns"),
        *[pl.col(c).cast(pl.Float64, strict=False) for c in THETA_COLS]
    ]).sort("t_label_ns")

    # Materialize to perform filtering and joins
    s_df = s.collect()
    l_df = l.collect()

    # --- Filter Invalid Labels ---
    # Only require valid timestamp. Individual theta NaN values are handled per-column
    # during interpolation (NaN in one theta doesn't discard other thetas).
    # Replace infinite values with null for proper handling.
    
    clean_exprs = []
    for c in THETA_COLS:
        if c in l_df.columns:
            clean_exprs.append(
                pl.when(pl.col(c).is_finite())
                .then(pl.col(c))
                .otherwise(None)
                .alias(c)
            )
    if clean_exprs:
        l_df = l_df.with_columns(clean_exprs)
    
    # Only require valid timestamp
    l_df = l_df.filter(pl.col("t_label_ns").is_not_null())

    if l_df.height < 2:
        raise RuntimeError(f"Too few valid label frames found (height={l_df.height}). Check labels.csv content.")

    # Report per-column coverage
    print(f"  - Total Label Frames: {l_df.height}")
    for c in THETA_COLS:
        if c in l_df.columns:
            valid_count = l_df.filter(pl.col(c).is_not_null()).height
            pct = valid_count / l_df.height * 100
            if pct < 100:
                print(f"    {c}: {valid_count}/{l_df.height} valid ({pct:.1f}%)")
    print(f"  - Sensor Samples: {s_df.height}")

    # --- Choose sensor time base ---
    # NOTE: We MUST use t_pc_ns (system time from PC) for merging with video labels
    # t_pc_ns is captured when record.py receives data, synchronized with video frames
    time_col = "t_pc_ns"  # Always use PC time for synchronization
    print(f"  - Using time column for merge: {time_col} (system-synchronized)")

    # --- Interpolation Strategy ---
    # We want to assign a theta vector to every sensor sample.
    # Since sensor rate (~1000Hz?) >> video rate (30-60Hz), we interpolate angles.
    
    # 1. Join Backward (Find previous frame t0 <= t_sensor)
    # Rename label cols to exclude them from collision or clarify source
    l_backward = l_df.select(["t_label_ns"] + THETA_COLS)
    
    merged = s_df.join_asof(
        l_backward,
        left_on=time_col,
        right_on="t_label_ns",
        strategy="backward"
    ).rename({c: f"{c}_prev" for c in THETA_COLS}).rename({"t_label_ns": "t_prev_ns"})

    # 2. Join Forward (Find next frame t1 >= t_sensor)
    l_forward = l_df.select(["t_label_ns"] + THETA_COLS)
    
    merged = merged.join_asof(
        l_forward,
        left_on=time_col,
        right_on="t_label_ns",
        strategy="forward"
    ).rename({c: f"{c}_next" for c in THETA_COLS}).rename({"t_label_ns": "t_next_ns"})

    # 3. Calculate Interpolation Weight
    # w = (t_sensor - t_prev) / (t_next - t_prev)
    # If t_next == t_prev, we avoid div by zero
    
    denom = (pl.col("t_next_ns") - pl.col("t_prev_ns")).cast(pl.Float64)
    weight = (pl.col(time_col) - pl.col("t_prev_ns")).cast(pl.Float64) / denom
    
    expressions = []
    
    # Interpolate each theta
    for c in THETA_COLS:
        prev_col = f"{c}_prev"
        next_col = f"{c}_next"
        
        # Linear Interp: prev + w * (next - prev)
        expr = (
            pl.when(denom > 0)
            .then(pl.col(prev_col) + weight * (pl.col(next_col) - pl.col(prev_col)))
            .otherwise(pl.col(prev_col)) # Fallback if exact match or single point
            .alias(c)
        )
        expressions.append(expr)
    
    merged = merged.with_columns(expressions)

    # 4. Filter Range (drop_outside)
    # If selected, we remove sensor data that happened before the first video frame or after the last
    if drop_outside:
        t_start = l_df["t_label_ns"].min()
        t_end = l_df["t_label_ns"].max()
        merged = merged.filter(
            (pl.col(time_col) >= t_start) & 
            (pl.col(time_col) <= t_end)
        )

    # 5. Cleanup
    # Keep standard columns: t_ns, sensor_id, raw data, interpolated angles
    keep_cols = ["t_pc_ns", "sensor_id", "accX", "accY", "accZ", "magX", "magY", "magZ"] + THETA_COLS
    final_df = merged.select([c for c in keep_cols if c in merged.columns])

    # 6. Save
    final_df.write_csv(out_path)
    print(f"[Merge] Successfully wrote {final_df.height} rows to {out_path}")
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Path to the run directory (containing sensors.csv and labels.csv)")
    ap.add_argument("--output", default="merged.csv", help="Output filename")
    ap.add_argument("--keep-all-time", action="store_true", 
                    help="If set, keeps sensor data outside the video timeline (extrapolated or null). Default is to drop.")
    
    args = ap.parse_args()
    
    run_path = Path(args.run_dir)
    if not run_path.exists():
        print(f"Error: Directory {run_path} does not exist.")
        return

    try:
        merge_run(run_path, args.output, drop_outside=not args.keep_all_time)
    except Exception as e:
        print(f"Error during merge: {e}")

if __name__ == "__main__":
    main()
