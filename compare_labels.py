import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def compare_labels(file1, file2):
    """
    Vergleicht zwei labels.csv Dateien und zeigt Unterschiede an.
    """
    print(f"Comparing:")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")
    print("-" * 60)
    
    # Lade CSVs
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    print(f"\nFile 1: {len(df1)} rows, {len(df1.columns)} columns")
    print(f"File 2: {len(df2)} rows, {len(df2.columns)} columns")
    
    # Finde gemeinsame Frame-Indizes
    common_frames = set(df1['frame_idx']).intersection(set(df2['frame_idx']))
    print(f"Common frames: {len(common_frames)}")
    
    # Sortiere beide nach frame_idx und filtere auf gemeinsame Frames
    df1_common = df1[df1['frame_idx'].isin(common_frames)].sort_values('frame_idx').reset_index(drop=True)
    df2_common = df2[df2['frame_idx'].isin(common_frames)].sort_values('frame_idx').reset_index(drop=True)
    
    # Finde Winkel-Spalten (theta_X)
    angle_cols = [col for col in df1.columns if col.startswith('theta_')]
    print(f"\nAngle columns found: {angle_cols}")
    
    # Vergleiche jeden Winkel
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON")
    print("="*60)
    
    for col in angle_cols:
        vals1 = df1_common[col].values
        vals2 = df2_common[col].values
        
        # Entferne NaN-Werte für Vergleich
        valid_mask = ~(np.isnan(vals1) | np.isnan(vals2))
        vals1_valid = vals1[valid_mask]
        vals2_valid = vals2[valid_mask]
        
        if len(vals1_valid) == 0:
            print(f"\n{col}: No valid data in both files")
            continue
        
        # Berechne Unterschiede
        diff = vals1_valid - vals2_valid
        abs_diff = np.abs(diff)
        
        print(f"\n{col}:")
        print(f"  Valid samples: {len(vals1_valid)}/{len(vals1)}")
        print(f"  Mean difference: {np.mean(diff):.4f}°")
        print(f"  Std deviation:   {np.std(diff):.4f}°")
        print(f"  Max difference:  {np.max(abs_diff):.4f}°")
        print(f"  Median diff:     {np.median(abs_diff):.4f}°")
        print(f"  95th percentile: {np.percentile(abs_diff, 95):.4f}°")
        
        # Zähle "signifikante" Unterschiede (>1°)
        large_diffs = np.sum(abs_diff > 1.0)
        if large_diffs > 0:
            print(f"  ⚠️  Differences >1°: {large_diffs} ({100*large_diffs/len(vals1_valid):.1f}%)")
    
    # Visualisierung
    print("\n" + "="*60)
    print("Generating comparison plots...")
    
    num_angles = len(angle_cols)
    fig, axes = plt.subplots(num_angles, 2, figsize=(14, 4*num_angles))
    if num_angles == 1:
        axes = axes.reshape(1, -1)
    
    for idx, col in enumerate(angle_cols):
        vals1 = df1_common[col].values
        vals2 = df2_common[col].values
        
        # Entferne NaN
        valid_mask = ~(np.isnan(vals1) | np.isnan(vals2))
        vals1_valid = vals1[valid_mask]
        vals2_valid = vals2[valid_mask]
        frames_valid = df1_common['frame_idx'].values[valid_mask]
        
        if len(vals1_valid) == 0:
            continue
        
        # Plot 1: Zeitreihe (erste 500 Frames)
        ax1 = axes[idx, 0]
        limit = min(500, len(frames_valid))
        ax1.plot(frames_valid[:limit], vals1_valid[:limit], label='Original', alpha=0.7, linewidth=1)
        ax1.plot(frames_valid[:limit], vals2_valid[:limit], label='Optimized', alpha=0.7, linewidth=1)
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel(f'{col} [°]')
        ax1.set_title(f'{col} - Time Series (first {limit} frames)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Differenz-Histogramm
        ax2 = axes[idx, 1]
        diff = vals1_valid - vals2_valid
        ax2.hist(diff, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
        ax2.set_xlabel('Difference [°]')
        ax2.set_ylabel('Count')
        ax2.set_title(f'{col} - Difference Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Stats in plot
        stats_text = f'Mean: {np.mean(diff):.3f}°\nStd: {np.std(diff):.3f}°\nMax: {np.max(np.abs(diff)):.3f}°'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Speichere Plot
    output_file = Path(file1).parent / "labels_comparison.png"
    plt.savefig(output_file, dpi=150)
    print(f"Comparison plot saved to: {output_file}")
    plt.show()
    
    # Zusammenfassung
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_diffs = []
    for col in angle_cols:
        vals1 = df1_common[col].values
        vals2 = df2_common[col].values
        valid_mask = ~(np.isnan(vals1) | np.isnan(vals2))
        if np.sum(valid_mask) > 0:
            diff = np.abs(vals1[valid_mask] - vals2[valid_mask])
            all_diffs.extend(diff)
    
    if len(all_diffs) > 0:
        all_diffs = np.array(all_diffs)
        print(f"Overall mean absolute difference: {np.mean(all_diffs):.4f}°")
        print(f"Overall max difference:           {np.max(all_diffs):.4f}°")
        print(f"Samples with diff > 0.5°:         {np.sum(all_diffs > 0.5)} ({100*np.sum(all_diffs > 0.5)/len(all_diffs):.1f}%)")
        print(f"Samples with diff > 1.0°:         {np.sum(all_diffs > 1.0)} ({100*np.sum(all_diffs > 1.0)/len(all_diffs):.1f}%)")
        
        if np.mean(all_diffs) < 0.5:
            print("\n✅ Excellent agreement! The optimized version produces very similar results.")
        elif np.mean(all_diffs) < 1.0:
            print("\n✓ Good agreement. Small differences are acceptable.")
        else:
            print("\n⚠️  Notable differences detected. Review the plots.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
    else:
        # Interactive mode
        print("Label Comparison Tool")
        print("Enter paths to two labels.csv files:")
        file1 = input("File 1 (original): ").strip()
        file2 = input("File 2 (optimized): ").strip()
    
    compare_labels(file1, file2)
