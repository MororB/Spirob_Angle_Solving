import argparse
from pathlib import Path
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

#python .\plot_sensors.py --run .\runs\2025-12-08_15-44-36\ --source merged --board 0 --sensor 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help=r"Run-Ordner, z.B. .\runs\2025-12-08_15-44-36\ ")
    ap.add_argument("--source", choices=["sensors", "merged"], default="merged",
                    help="merged nutzt die gleiche Zeitbasis wie Thetas; sensors ist Rohlog.")
    ap.add_argument("--board", type=int, default=0, help="board_id (0..2)")
    ap.add_argument("--sensor", type=int, default=0, help="sensor_id (0..4)")
    ap.add_argument("--every", type=int, default=1, help="Downsampling (jeden N-ten Punkt)")
    ap.add_argument("--out", default=None, help="Optional PNG speichern")
    args = ap.parse_args()

    run_dir = Path(args.run)
    csv_path = run_dir / ("merged.csv" if args.source == "merged" else "sensors.csv")
    if not csv_path.exists():
        raise SystemExit(f"Nicht gefunden: {csv_path}")

    df = pl.read_csv(csv_path, infer_schema_length=5000)

    # Zeitspalte wählen
    t_col = "t_ns" if "t_ns" in df.columns else "t_frame_ns"
    df = df.with_columns([
        pl.col(t_col).cast(pl.Int64, strict=False),
        pl.col("board_id").cast(pl.Int64, strict=False) if "board_id" in df.columns else pl.lit(None),
        pl.col("sensor_id").cast(pl.Int64, strict=False) if "sensor_id" in df.columns else pl.lit(None),
        pl.col("x").cast(pl.Float64, strict=False),
        pl.col("y").cast(pl.Float64, strict=False),
        pl.col("z").cast(pl.Float64, strict=False),
    ]).drop_nulls([t_col]).sort(t_col)

    # nach board/sensor filtern
    if "board_id" in df.columns and "sensor_id" in df.columns:
        df = df.filter((pl.col("board_id") == args.board) & (pl.col("sensor_id") == args.sensor))

    if df.height < 5:
        raise SystemExit("Zu wenige Daten nach Filter. Prüfe board/sensor oder Datei.")

    # Zeit in Sekunden ab Start
    t = df[t_col].to_numpy()
    t0 = t[0]
    x_time = (t - t0) / 1e9

    # Downsample
    step = max(1, args.every)
    x_time = x_time[::step]

    bx = df["x"].to_numpy()[::step]
    by = df["y"].to_numpy()[::step]
    bz = df["z"].to_numpy()[::step]
    bmag = np.sqrt(bx*bx + by*by + bz*bz)

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 9))
    title = f"{args.source}.csv | board={args.board} sensor={args.sensor} | {csv_path}"
    fig.suptitle(title)

    axs[0].plot(x_time, bx); axs[0].set_ylabel("Bx")
    axs[1].plot(x_time, by); axs[1].set_ylabel("By")
    axs[2].plot(x_time, bz); axs[2].set_ylabel("Bz")
    axs[3].plot(x_time, bmag); axs[3].set_ylabel("|B|")

    for ax in axs:
        ax.grid(True)
    axs[-1].set_xlabel("Zeit [s]")

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=150)
        print("Gespeichert:", args.out)
    plt.show()

if __name__ == "__main__":
    main()
