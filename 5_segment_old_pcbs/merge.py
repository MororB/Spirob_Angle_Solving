import argparse
from pathlib import Path
import polars as pl
import math

THETA_COLS = ["theta_1", "theta_2", "theta_3", "theta_4"]

def merge_run(run_dir: Path, out_csv_name: str = "merged.csv", drop_outside: bool = False) -> Path:
    sensors_path = run_dir / "sensors.csv"
    labels_path  = run_dir / "labels.csv"
    out_path     = run_dir / out_csv_name

    # 1. Daten einlesen (Polars LazyFrame für Optimierung)
    s = pl.scan_csv(sensors_path, infer_schema_length=5000).sort("t_ns")
    l = pl.scan_csv(labels_path,  infer_schema_length=5000).sort("t_frame_ns")
    
    # Sofortiger Stop, wenn Dateien fehlen
    if not sensors_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Missing sensors.csv or labels.csv in {run_dir}")

    # --- 2. Label-Daten bereinigen (Schema-Enforcement und Filtern) ---
    
    # 2a. Typen robust machen
    l = l.with_columns([
        # Timesatamps (sollten als Int64 vorliegen)
        pl.col("t_frame_ns").cast(pl.Int64, strict=False),
        # Quality (falls fehlt oder falsch)
        pl.col("quality").cast(pl.Int64, strict=False).fill_null(1).alias("quality")
    ])

    # 2b. Thetas: Casten von String ('nan') zu Float (NaN). strict=False setzt nicht-numerische Werte auf Null.
    l = l.with_columns([
        pl.col(c).cast(pl.Float64, strict=False) for c in THETA_COLS
    ])

    # 2c. Effizientes Filtern (nur eine Kopie des DataFrames)
    # Nur gute Frames (quality=1) und alle Theta-Spalten müssen finit und nicht-null sein
    theta_filter_expr = pl.lit(True)
    for c in THETA_COLS:
        theta_filter_expr = theta_filter_expr & pl.col(c).is_not_null() & pl.col(c).is_finite()
        
    l = l.filter(
        (pl.col("quality") == 1) & theta_filter_expr
    ).collect() # Jetzt materialisieren (vom LazyFrame zum DataFrame)
    
    if l.height < 2:
        raise RuntimeError("Zu wenige gültige Label-Frames (quality==1 + finite thetas).")

    # --- 3. Zeitfenster und Sensor-Daten vorbereiten ---
    t_min = l["t_frame_ns"].min()
    t_max = l["t_frame_ns"].max()

    s = s.with_columns([
        pl.col("t_ns").cast(pl.Int64, strict=False),
        ((pl.col("t_ns") >= t_min) & (pl.col("t_ns") <= t_max)).alias("in_label_time")
    ]).collect() # Materialisieren der Sensordaten

    # --- 4. Join Asof (Backward und Forward) ---
    
    # Backward Join (t0)
    m = s.join_asof(
        l.select(["t_frame_ns", *THETA_COLS]),
        left_on="t_ns",
        right_on="t_frame_ns",
        strategy="backward",
    )

    # Forward Join (t1)
    l_fwd = (
        l.select(["t_frame_ns", *THETA_COLS])
         .rename({
             "t_frame_ns":"t_frame_ns_fwd", 
             **{c:f"{c}_fwd" for c in THETA_COLS}
         })
    )

    m = m.join_asof(
        l_fwd,
        left_on="t_ns",
        right_on="t_frame_ns_fwd",
        strategy="forward",
    )

    # --- 5. Lineare Interpolation ---
    # Nenner: Differenz zwischen t1 und t0 (sollte immer positiv sein, da l sortiert ist)
    denom = (pl.col("t_frame_ns_fwd") - pl.col("t_frame_ns")).cast(pl.Float64)
    # Gewicht w: (t_ns - t0) / (t1 - t0)
    w = (pl.col("t_ns") - pl.col("t_frame_ns")).cast(pl.Float64) / denom

    for c in THETA_COLS:
        c0 = pl.col(c)       # Theta bei t0 (backward join)
        c1 = pl.col(f"{c}_fwd") # Theta bei t1 (forward join)
        
        # Interpolationsformel: theta = c0 + w * (c1 - c0)
        # Überprüfung auf denom != 0 ist wichtig für numerische Stabilität
        interp = (
            pl.when(denom.is_finite() & (denom != 0))
              .then(c0 + w * (c1 - c0))
              .otherwise(None) # Wenn t0 == t1 oder Randfall, setze Null
              .alias(c) # Wichtig: die originalen Theta-Spalten überschreiben
        )
        m = m.with_columns(interp)

    # --- 6. Aufräumen und Speichern ---
    if drop_outside:
        # Nur Sensor-Daten verwenden, die im Zeitbereich der Frames liegen
        m = m.filter(pl.col("in_label_time") == True)
        
    # Die temporären Spalten (t1, theta_fwd) löschen
    drop_cols = ["t_frame_ns", "t_frame_ns_fwd", "in_label_time"] + [f"{c}_fwd" for c in THETA_COLS]
    
    m = m.drop([c for c in drop_cols if c in m.columns])

    m.write_csv(out_path)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Der Pfad zum Run-Ordner (z.B. runs/2025-12-08_15-00-00)")
    ap.add_argument("--drop-outside", action="store_true", 
                    help="Nur Sensor-Daten verwenden, die zeitlich zwischen dem ersten und letzten Label-Frame liegen.")
    args = ap.parse_args()

    try:
        out = merge_run(Path(args.run), drop_outside=args.drop_outside)
        print("Geschrieben:", out)
    except Exception as e:
        print("FEHLER beim Mergen:", e)

if __name__ == "__main__":
    main()