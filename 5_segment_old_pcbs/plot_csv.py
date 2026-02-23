import argparse
from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt

#python .\plot_csv.py --run .\runs\2025-12-08_13-41-29\
#python .\plot_csv.py --run .\runs\2025-12-08_13-41-29\ --source merged
THETA_COLS = ["theta_1", "theta_2", "theta_3", "theta_4"]


def load_labels(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=5000)
    # robust cast
    cols = ["t_frame_ns"] + (["quality"] if "quality" in df.columns else [])
    df = df.with_columns([pl.col(c).cast(pl.Int64, strict=False) for c in cols if c in df.columns])
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in THETA_COLS if c in df.columns])
    df = df.drop_nulls(["t_frame_ns"]).sort("t_frame_ns")
    return df


def load_merged(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=5000)
    df = df.with_columns([pl.col("t_ns").cast(pl.Int64, strict=False)])
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in THETA_COLS if c in df.columns])
    df = df.drop_nulls(["t_ns"]).sort("t_ns")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help=r"Run-Ordner, z.B. .\runs\2025-12-08_13-41-29\ ")
    ap.add_argument("--file", help="Direkter Pfad zu einer CSV (labels.csv oder merged.csv)")
    ap.add_argument("--source", choices=["labels", "merged"], default=None,
                    help="Optional: erzwinge Quelle (labels oder merged)")
    ap.add_argument("--every", type=int, default=1, help="Nur jeden N-ten Punkt plotten (Downsampling)")
    ap.add_argument("--out", default=None, help="Optional: als PNG speichern (z.B. plot.png)")
    args = ap.parse_args()

    if args.file:
        csv_path = Path(args.file)
    elif args.run:
        run_dir = Path(args.run)
        # default: labels
        if args.source == "merged":
            csv_path = run_dir / "merged.csv"
        else:
            csv_path = run_dir / "labels.csv"
    else:
        raise SystemExit("Bitte --run oder --file angeben.")

    if not csv_path.exists():
        raise SystemExit(f"Datei nicht gefunden: {csv_path}")

    # Quelle erkennen/erzwingen
    source = args.source
    if source is None:
        source = "labels" if "labels" in csv_path.name.lower() else "merged"

    if source == "labels":
        df = load_labels(csv_path)
        t = df["t_frame_ns"].to_numpy()
        t0 = t[0]
        x = (t - t0) / 1e9  # Sekunden seit Start
        title = f"Labels: {csv_path}"
        quality = df["quality"].to_numpy() if "quality" in df.columns else None
    else:
        df = load_merged(csv_path)
        t = df["t_ns"].to_numpy()
        t0 = t[0]
        x = (t - t0) / 1e9  # Sekunden seit Start
        title = f"Merged: {csv_path}"
        quality = df["in_label_time"].to_numpy() if "in_label_time" in df.columns else None

    # Downsample
    step = max(1, args.every)
    x = x[::step]

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 9))
    fig.suptitle(title)

    for i, c in enumerate(THETA_COLS):
        if c not in df.columns:
            axs[i].text(0.1, 0.5, f"{c} fehlt", transform=axs[i].transAxes)
            continue
        y = df[c].to_numpy()[::step]
        axs[i].plot(x, y)
        axs[i].set_ylabel(f"{c} [deg]")  # Einheit = so wie gespeichert
        axs[i].grid(True)

    axs[-1].set_xlabel("Zeit [s]")

    # optional quality anzeigen (als Punkte oben drauf)
    if quality is not None:
        q = quality[::step]
        # skaliere auf Achsenhöhe, nur als Visualisierung
        for ax in axs:
            ymin, ymax = ax.get_ylim()
            yq = ymin + 0.95 * (ymax - ymin) * (q.astype(float))
            ax.plot(x, yq, ".", markersize=2)

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=150)
        print("Gespeichert:", args.out)
    plt.show()


if __name__ == "__main__":
    main()
