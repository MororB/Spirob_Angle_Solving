"""
Smooth Joint Angles in Merged CSV Files

Dieses Skript smootht die theta-Winkel in merged.csv Dateien,
um ArUco-Tracking-Jitter zu reduzieren.

Erstellt neue Dateien: merged_smoothed.csv
"""

import polars as pl
import numpy as np
from scipy.signal import savgol_filter
from pathlib import Path
import argparse

# ==================== KONFIGURATION ====================

# Runs zum Verarbeiten
RUNS = [
    #"runs/2026-02-19_10-14-37",
    "runs/2026-02-05_14-46-22",
    "runs/2026-02-06_14-14-43",
    "runs/2026-02-09_12-34-53",
    "runs/2026-02-10_12-14-57",
    "runs/2026-02-11_17-22-11",

    ]

# Smoothing-Methoden: "savgol", "moving_average", "gaussian"
SMOOTHING_METHOD = "gaussian"

# Savitzky-Golay Filter Parameter
SAVGOL_WINDOW_LENGTH = 21  # Ungerade Zahl! (empfohlen: 11-51)
SAVGOL_POLYORDER = 3       # Polynom-Grad (empfohlen: 2-5)

# Moving Average Parameter
MA_WINDOW_SIZE = 15        # Fenstergröße für gleitenden Durchschnitt

# Gaussian Filter Parameter
GAUSSIAN_SIGMA = 3.0       # Standardabweichung für Gauß-Filter


# ==================== SMOOTHING FUNKTIONEN ====================

def smooth_savgol(data, window_length=21, polyorder=3):
    """
    Savitzky-Golay Filter - Sehr gut für Beibehaltung von Peaks
    """
    if len(data) < window_length:
        print(f"  ⚠️  Warnung: Daten zu kurz für window_length={window_length}, nutze {len(data)//2*2-1}")
        window_length = max(polyorder + 2, len(data) // 2 * 2 - 1)
    
    if window_length % 2 == 0:
        window_length += 1  # Muss ungerade sein
    
    return savgol_filter(data, window_length, polyorder)


def smooth_moving_average(data, window_size=15):
    """
    Gleitender Durchschnitt - Einfach und effektiv
    """
    cumsum = np.cumsum(np.insert(data, 0, 0))
    result = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    # Padding am Anfang/Ende
    pad_start = np.full(window_size // 2, result[0])
    pad_end = np.full(len(data) - len(result) - len(pad_start), result[-1])
    
    return np.concatenate([pad_start, result, pad_end])


def smooth_gaussian(data, sigma=3.0):
    """
    Gauß-Filter - Sehr sanftes Smoothing
    """
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(data, sigma)


# ==================== HAUPTFUNKTION ====================

def smooth_merged_csv(input_path, output_path, method="savgol", **kwargs):
    """
    Smootht theta-Spalten in einer merged.csv Datei
    
    Args:
        input_path: Pfad zur Eingabedatei
        output_path: Pfad zur Ausgabedatei
        method: "savgol", "moving_average", oder "gaussian"
        **kwargs: Parameter für die Smoothing-Methode
    """
    print(f"\n📂 Verarbeite: {input_path}")
    
    # Lade Daten
    df = pl.read_csv(input_path)
    
    # Finde theta-Spalten
    theta_cols = [c for c in df.columns if c.startswith("theta_")]
    
    if not theta_cols:
        print(f"  ⚠️  Keine theta-Spalten gefunden!")
        return
    
    print(f"  📊 Gefunden: {len(theta_cols)} Winkel-Spalten")
    print(f"  🔧 Methode: {method}")
    
    # Konvertiere zu pandas für einfachere Manipulation
    df_pd = df.to_pandas()
    
    # Smoothe jede theta-Spalte
    for col in theta_cols:
        original = df_pd[col].values
        
        # Wähle Smoothing-Methode
        if method == "savgol":
            window_length = kwargs.get("window_length", SAVGOL_WINDOW_LENGTH)
            polyorder = kwargs.get("polyorder", SAVGOL_POLYORDER)
            smoothed = smooth_savgol(original, window_length, polyorder)
            
        elif method == "moving_average":
            window_size = kwargs.get("window_size", MA_WINDOW_SIZE)
            smoothed = smooth_moving_average(original, window_size)
            
        elif method == "gaussian":
            sigma = kwargs.get("sigma", GAUSSIAN_SIGMA)
            smoothed = smooth_gaussian(original, sigma)
            
        else:
            raise ValueError(f"Unbekannte Methode: {method}")
        
        # Ersetze Original-Werte
        df_pd[col] = smoothed
        
        # Berechne Verbesserung
        original_std = np.std(np.diff(original))
        smoothed_std = np.std(np.diff(smoothed))
        reduction = (1 - smoothed_std / original_std) * 100
        
        print(f"    {col}: Jitter reduziert um {reduction:.1f}%")
    
    # Speichere als neue Datei
    df_smoothed = pl.from_pandas(df_pd)
    df_smoothed.write_csv(output_path)
    
    print(f"  ✅ Gespeichert: {output_path}")
    
    return df_smoothed


# ==================== BATCH-VERARBEITUNG ====================

def process_all_runs(runs=RUNS, method=SMOOTHING_METHOD, **kwargs):
    """
    Verarbeitet alle Runs
    """
    print("=" * 60)
    print("🔧 WINKEL SMOOTHING - Batch Processing")
    print("=" * 60)
    print(f"Methode: {method}")
    
    if method == "savgol":
        print(f"  Window: {kwargs.get('window_length', SAVGOL_WINDOW_LENGTH)}")
        print(f"  Polyorder: {kwargs.get('polyorder', SAVGOL_POLYORDER)}")
    elif method == "moving_average":
        print(f"  Window Size: {kwargs.get('window_size', MA_WINDOW_SIZE)}")
    elif method == "gaussian":
        print(f"  Sigma: {kwargs.get('sigma', GAUSSIAN_SIGMA)}")
    
    success_count = 0
    
    for run_path in runs:
        input_file = Path(run_path) / "merged.csv"
        output_file = Path(run_path) / "merged_smoothed.csv"
        
        if not input_file.exists():
            print(f"\n⚠️  Überspringe {run_path}: merged.csv nicht gefunden")
            continue
        
        try:
            smooth_merged_csv(str(input_file), str(output_file), method, **kwargs)
            success_count += 1
        except Exception as e:
            print(f"\n❌ Fehler bei {run_path}: {e}")
    
    print("\n" + "=" * 60)
    print(f"✅ Fertig! {success_count}/{len(runs)} Dateien verarbeitet")
    print("=" * 60)


# ==================== MAIN ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smooth joint angles in merged.csv files"
    )
    
    parser.add_argument(
        "--method", 
        choices=["savgol", "moving_average", "gaussian"],
        default=SMOOTHING_METHOD,
        help="Smoothing-Methode"
    )
    
    parser.add_argument(
        "--window", 
        type=int, 
        default=None,
        help="Window Length (savgol/moving_average)"
    )
    
    parser.add_argument(
        "--poly", 
        type=int, 
        default=SAVGOL_POLYORDER,
        help="Polynomial Order (nur savgol)"
    )
    
    parser.add_argument(
        "--sigma", 
        type=float, 
        default=GAUSSIAN_SIGMA,
        help="Sigma (nur gaussian)"
    )
    
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Verarbeite nur einen einzelnen Run (Pfad)"
    )
    
    args = parser.parse_args()
    
    # Prepare kwargs
    kwargs = {}
    
    if args.method == "savgol":
        kwargs["window_length"] = args.window if args.window else SAVGOL_WINDOW_LENGTH
        kwargs["polyorder"] = args.poly
    elif args.method == "moving_average":
        kwargs["window_size"] = args.window if args.window else MA_WINDOW_SIZE
    elif args.method == "gaussian":
        kwargs["sigma"] = args.sigma
    
    # Process
    if args.single:
        input_file = Path(args.single) / "merged.csv"
        output_file = Path(args.single) / "merged_smoothed.csv"
        smooth_merged_csv(str(input_file), str(output_file), args.method, **kwargs)
    else:
        process_all_runs(RUNS, args.method, **kwargs)
