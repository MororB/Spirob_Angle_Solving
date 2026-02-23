"""
Dual-Magnet Magnetfeld-Simulation mit Magpylib
=================================================
Simuliert ZWEI Zylindermagneten, die pendelnd um je einen Drehpunkt schwingen,
und berechnet das überlagerte B-Feld an einem fixen Sensor für verschiedene
Magnetausrichtungs-Kombinationen.

Geometrie:
  - Sensor am Ursprung (0, 0, 0)
  - Magnet 1 bei (+4.0 cm, 0, 0), Drehpunkt bei (+2.0 cm, 0, 0)
  - Magnet 2 bei (-3.5 cm, 0, 0), Drehpunkt bei (-1.75 cm, 0, 0)
    (leicht näher am Sensor als Magnet 1)

Beide Magnete schwingen als Pendel um die z-Achse mit jeweiligem Drehpunkt.
Das B-Feld am Sensor ist die Superposition beider Einzelfelder.

Einheiten: SI (Meter, Tesla, A/m)
"""

import numpy as np
import magpylib as magpy
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ==============================================================
# KONFIGURATION
# ==============================================================

# Magnet-Parameter (Zylinder, identisch für beide)
DIAMETER = 0.01          # 1 cm Durchmesser
HEIGHT = 0.01            # 1 cm Höhe
MAG_VECTOR = (0, 0, 1e6)  # Magnetisierung in A/m (J ≈ µ0*M ≈ 1.26 T)

# Geometrie (SI: Meter)
SENSOR_POS = np.array([0.0, 0.0, 0.0])

# Magnet 1: rechts vom Sensor (+x)
MAGNET1_INIT_POS = np.array([0.04, 0.0, 0.0])    # 4.0 cm in +x
PIVOT1 = np.array([0.02, 0.0, 0.0])               # Drehpunkt bei 2.0 cm

# Magnet 2: links vom Sensor (-x), leicht näher (3.5 cm statt 4.0 cm)
MAGNET2_INIT_POS = np.array([-0.035, 0.0, 0.0])   # 3.5 cm in -x
PIVOT2 = np.array([-0.0175, 0.0, 0.0])            # Drehpunkt bei -1.75 cm

# Pfad-Parameter
SWING_ANGLE = 30   # ±30 Grad Schwingung um z-Achse
N_HALF = 50        # Punkte pro Halbschwingung
N_PATH = 2 * N_HALF  # 100 Punkte gesamt

# Orientierungen für Magnet 1 (erweiterte Liste)
M1_ORIENTATIONS = [
    # Hauptachsen (6 Orientierungen)
    ("M1_+z",     None),
    ("M1_+x",    R.from_euler("y", 90, degrees=True)),
    ("M1_+y",    R.from_euler("x", 90, degrees=True)),
    ("M1_-z",    R.from_euler("y", 180, degrees=True)),
    ("M1_-x",    R.from_euler("y", -90, degrees=True)),
    ("M1_-y",    R.from_euler("x", -90, degrees=True)),
    
    # 30° Rotationen um einzelne Achsen (6 Orientierungen)
    ("M1_30x",   R.from_euler("x", 30, degrees=True)),
    ("M1_30y",   R.from_euler("y", 30, degrees=True)),
    ("M1_30z",   R.from_euler("z", 30, degrees=True)),
    ("M1_-30x",  R.from_euler("x", -30, degrees=True)),
    ("M1_-30y",  R.from_euler("y", -30, degrees=True)),
    ("M1_-30z",  R.from_euler("z", -30, degrees=True)),
    
    # 45° Rotationen um einzelne Achsen (3 Orientierungen)
    ("M1_45x",   R.from_euler("x", 45, degrees=True)),
    ("M1_45y",   R.from_euler("y", 45, degrees=True)),
    ("M1_45z",   R.from_euler("z", 45, degrees=True)),
    
    # 60° Rotationen um einzelne Achsen (6 Orientierungen)
    ("M1_60x",   R.from_euler("x", 60, degrees=True)),
    ("M1_60y",   R.from_euler("y", 60, degrees=True)),
    ("M1_60z",   R.from_euler("z", 60, degrees=True)),
    ("M1_-60x",  R.from_euler("x", -60, degrees=True)),
    ("M1_-60y",  R.from_euler("y", -60, degrees=True)),
    ("M1_-60z",  R.from_euler("z", -60, degrees=True)),
    
    # Kombinierte Rotationen 30° (6 Orientierungen)
    ("M1_30xy",  R.from_euler("xy", [30, 30], degrees=True)),
    ("M1_30xz",  R.from_euler("xz", [30, 30], degrees=True)),
    ("M1_30yz",  R.from_euler("yz", [30, 30], degrees=True)),
    ("M1_-30xy", R.from_euler("xy", [-30, -30], degrees=True)),
    ("M1_-30xz", R.from_euler("xz", [-30, -30], degrees=True)),
    ("M1_-30yz", R.from_euler("yz", [-30, -30], degrees=True)),
    
    # Kombinierte Rotationen 45° (3 Orientierungen)
    ("M1_45xy",  R.from_euler("xy", [45, 45], degrees=True)),
    ("M1_45xz",  R.from_euler("xz", [45, 45], degrees=True)),
    ("M1_45yz",  R.from_euler("yz", [45, 45], degrees=True)),
    
    # Kombinierte Rotationen 60° (6 Orientierungen)
    ("M1_60xy",  R.from_euler("xy", [60, 60], degrees=True)),
    ("M1_60xz",  R.from_euler("xz", [60, 60], degrees=True)),
    ("M1_60yz",  R.from_euler("yz", [60, 60], degrees=True)),
    ("M1_-60xy", R.from_euler("xy", [-60, -60], degrees=True)),
    ("M1_-60xz", R.from_euler("xz", [-60, -60], degrees=True)),
    ("M1_-60yz", R.from_euler("yz", [-60, -60], degrees=True)),
    
    # Gemischte Winkel (6 Orientierungen)
    ("M1_30+60xy", R.from_euler("xy", [30, 60], degrees=True)),
    ("M1_30+60xz", R.from_euler("xz", [30, 60], degrees=True)),
    ("M1_30+60yz", R.from_euler("yz", [30, 60], degrees=True)),
    ("M1_30+45xy", R.from_euler("xy", [30, 45], degrees=True)),
    ("M1_30+45xz", R.from_euler("xz", [30, 45], degrees=True)),
    ("M1_45+60yz", R.from_euler("yz", [45, 60], degrees=True)),
]

# Orientierungen für Magnet 2 (erweiterte Liste)
M2_ORIENTATIONS = [
    # Hauptachsen (6 Orientierungen)
    ("M2_+z",     None),
    ("M2_+x",    R.from_euler("y", 90, degrees=True)),
    ("M2_+y",    R.from_euler("x", 90, degrees=True)),
    ("M2_-z",    R.from_euler("y", 180, degrees=True)),
    ("M2_-x",    R.from_euler("y", -90, degrees=True)),
    ("M2_-y",    R.from_euler("x", -90, degrees=True)),
    
    # 30° Rotationen um einzelne Achsen (6 Orientierungen)
    ("M2_30x",   R.from_euler("x", 30, degrees=True)),
    ("M2_30y",   R.from_euler("y", 30, degrees=True)),
    ("M2_30z",   R.from_euler("z", 30, degrees=True)),
    ("M2_-30x",  R.from_euler("x", -30, degrees=True)),
    ("M2_-30y",  R.from_euler("y", -30, degrees=True)),
    ("M2_-30z",  R.from_euler("z", -30, degrees=True)),
    
    # 45° Rotationen um einzelne Achsen (3 Orientierungen)
    ("M2_45x",   R.from_euler("x", 45, degrees=True)),
    ("M2_45y",   R.from_euler("y", 45, degrees=True)),
    ("M2_45z",   R.from_euler("z", 45, degrees=True)),
    
    # 60° Rotationen um einzelne Achsen (6 Orientierungen)
    ("M2_60x",   R.from_euler("x", 60, degrees=True)),
    ("M2_60y",   R.from_euler("y", 60, degrees=True)),
    ("M2_60z",   R.from_euler("z", 60, degrees=True)),
    ("M2_-60x",  R.from_euler("x", -60, degrees=True)),
    ("M2_-60y",  R.from_euler("y", -60, degrees=True)),
    ("M2_-60z",  R.from_euler("z", -60, degrees=True)),
    
    # Kombinierte Rotationen 30° (6 Orientierungen)
    ("M2_30xy",  R.from_euler("xy", [30, 30], degrees=True)),
    ("M2_30xz",  R.from_euler("xz", [30, 30], degrees=True)),
    ("M2_30yz",  R.from_euler("yz", [30, 30], degrees=True)),
    ("M2_-30xy", R.from_euler("xy", [-30, -30], degrees=True)),
    ("M2_-30xz", R.from_euler("xz", [-30, -30], degrees=True)),
    ("M2_-30yz", R.from_euler("yz", [-30, -30], degrees=True)),
    
    # Kombinierte Rotationen 45° (3 Orientierungen)
    ("M2_45xy",  R.from_euler("xy", [45, 45], degrees=True)),
    ("M2_45xz",  R.from_euler("xz", [45, 45], degrees=True)),
    ("M2_45yz",  R.from_euler("yz", [45, 45], degrees=True)),
    
    # Kombinierte Rotationen 60° (6 Orientierungen)
    ("M2_60xy",  R.from_euler("xy", [60, 60], degrees=True)),
    ("M2_60xz",  R.from_euler("xz", [60, 60], degrees=True)),
    ("M2_60yz",  R.from_euler("yz", [60, 60], degrees=True)),
    ("M2_-60xy", R.from_euler("xy", [-60, -60], degrees=True)),
    ("M2_-60xz", R.from_euler("xz", [-60, -60], degrees=True)),
    ("M2_-60yz", R.from_euler("yz", [-60, -60], degrees=True)),
    
    # Gemischte Winkel (6 Orientierungen)
    ("M2_30+60xy", R.from_euler("xy", [30, 60], degrees=True)),
    ("M2_30+60xz", R.from_euler("xz", [30, 60], degrees=True)),
    ("M2_30+60yz", R.from_euler("yz", [30, 60], degrees=True)),
    ("M2_30+45xy", R.from_euler("xy", [30, 45], degrees=True)),
    ("M2_30+45xz", R.from_euler("xz", [30, 45], degrees=True)),
    ("M2_45+60yz", R.from_euler("yz", [45, 60], degrees=True)),
]

# Alle Kombinationen generieren
DUAL_ORIENTATIONS: list[tuple[str, R | None, R | None]] = []
for m1_label, m1_rot in M1_ORIENTATIONS:
    for m2_label, m2_rot in M2_ORIENTATIONS:
        DUAL_ORIENTATIONS.append((f"{m1_label} | {m2_label}", m1_rot, m2_rot))


# ==============================================================
# HILFSFUNKTIONEN
# ==============================================================

def generate_pendulum_angles(n_half: int, swing_deg: float) -> np.ndarray:
    """Erzeugt Pendelwinkel: hin (-swing -> +swing) und zurück (+swing -> -swing)."""
    fwd = np.linspace(-swing_deg, swing_deg, n_half)
    bwd = np.linspace(swing_deg, -swing_deg, n_half)
    return np.concatenate([fwd, bwd])


def simulate_dual_orientation(
    label: str,
    orient_rot_m1: R | None,
    orient_rot_m2: R | None,
    angles: np.ndarray,
) -> dict:
    """
    Simuliert zwei Magnete für eine bestimmte Ausrichtungs-Kombination
    über den gesamten Pendelpfad.

    Beide Magnete schwingen unabhängig um ihre jeweiligen Drehpunkte.
    Das B-Feld am Sensor ist die Superposition (automatisch via Collection).

    Returns:
        dict mit keys: 'label', 'B', 'B1', 'B2',
                        'positions1', 'positions2', 'distances1', 'distances2',
                        'magnet1', 'magnet2', 'collection'
    """
    sensor = magpy.Sensor(position=SENSOR_POS)

    # --- Magnet 1 erstellen und konfigurieren ---
    magnet1 = magpy.magnet.Cylinder(
        dimension=(DIAMETER, HEIGHT),
        magnetization=MAG_VECTOR,
        position=MAGNET1_INIT_POS.copy(),
    )
    if orient_rot_m1 is not None:
        magnet1.rotate(orient_rot_m1, anchor=None)
    magnet1.rotate_from_angax(angle=angles, axis="z", anchor=PIVOT1)

    # --- Magnet 2 erstellen und konfigurieren ---
    magnet2 = magpy.magnet.Cylinder(
        dimension=(DIAMETER, HEIGHT),
        magnetization=MAG_VECTOR,
        position=MAGNET2_INIT_POS.copy(),
    )
    if orient_rot_m2 is not None:
        magnet2.rotate(orient_rot_m2, anchor=None)
    magnet2.rotate_from_angax(angle=angles, axis="z", anchor=PIVOT2)

    # --- B-Feld berechnen ---
    # Collection summiert automatisch die Felder beider Magnete
    coll = magpy.Collection(magnet1, magnet2)
    B_total = magpy.getB(coll, sensor)  # shape: (N_PATH+1, 3)

    # Einzelfelder für Analyse
    B1 = magpy.getB(magnet1, sensor)    # shape: (N_PATH+1, 3)
    B2 = magpy.getB(magnet2, sensor)    # shape: (N_PATH+1, 3)

    # Positionen und Abstände
    pos1 = magnet1.position
    pos2 = magnet2.position
    dist1 = np.linalg.norm(pos1 - SENSOR_POS, axis=1)
    dist2 = np.linalg.norm(pos2 - SENSOR_POS, axis=1)

    return {
        "label": label,
        "B": B_total,
        "B1": B1,
        "B2": B2,
        "positions1": pos1,
        "positions2": pos2,
        "distances1": dist1,
        "distances2": dist2,
        "magnet1": magnet1,
        "magnet2": magnet2,
        "collection": coll,
    }


def compute_info_gain(B: np.ndarray) -> dict:
    """
    Berechnet den Informationsgewinn über mehrere Metriken.

    Identisch zum Single-Magnet-Code – die Metriken sind unabhängig
    davon, ob das B-Feld von einem oder mehreren Magneten stammt.
    """
    B_norm = np.linalg.norm(B, axis=1)
    mean_norm = np.mean(B_norm)

    var_norm = np.var(B_norm, ddof=1)
    std_norm = np.std(B_norm, ddof=1)
    var_components = np.var(B, axis=0, ddof=1)
    std_components = np.std(B, axis=0, ddof=1)
    total_var = np.sum(var_components)

    cov_matrix = np.cov(B.T)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    det_cov = np.prod(eigenvalues)

    d = 3
    if det_cov > 0:
        entropy = 0.5 * (d * np.log(2 * np.pi * np.e) + np.log(det_cov))
    else:
        entropy = -np.inf

    if mean_norm > 1e-20 and det_cov > 0:
        cov_normalized = cov_matrix / (mean_norm ** 2)
        det_cov_norm = np.linalg.det(cov_normalized)
        if det_cov_norm > 0:
            entropy_normalized = 0.5 * (d * np.log(2 * np.pi * np.e) + np.log(det_cov_norm))
        else:
            entropy_normalized = -np.inf
    else:
        entropy_normalized = -np.inf

    snr = std_norm / mean_norm if mean_norm > 1e-20 else 0.0
    normalized_var = total_var / (mean_norm**2) if mean_norm > 1e-20 else 0.0

    lambda_min = eigenvalues[0]
    lambda_max = eigenvalues[-1]
    condition_num = lambda_max / lambda_min if lambda_min > 1e-30 else np.inf

    min_singular_value = np.sqrt(lambda_min)

    if np.all(eigenvalues > 1e-30):
        eig_sum = np.sum(eigenvalues)
        eig_prob = eigenvalues / eig_sum
        shannon_eig = -np.sum(eig_prob * np.log(eig_prob + 1e-30))
        effective_dim = np.exp(shannon_eig)
    else:
        effective_dim = 0.0

    geom_mean_eig = det_cov ** (1/3) if det_cov > 0 else 0.0
    arith_mean_eig = np.mean(eigenvalues)
    volumetric_efficiency = geom_mean_eig / arith_mean_eig if arith_mean_eig > 1e-30 else 0.0

    return {
        "var_norm": var_norm,
        "std_norm": std_norm,
        "var_components": var_components,
        "std_components": std_components,
        "total_var": total_var,
        "cov_matrix": cov_matrix,
        "eigenvalues": eigenvalues,
        "det_cov": det_cov,
        "entropy": entropy,
        "entropy_normalized": entropy_normalized,
        "snr": snr,
        "normalized_var": normalized_var,
        "condition_num": condition_num,
        "min_singular_value": min_singular_value,
        "effective_dim": effective_dim,
        "volumetric_efficiency": volumetric_efficiency,
    }


# ==============================================================
# PLAUSIBILITÄTSCHECKS
# ==============================================================

def plausibility_checks(results: list[dict]) -> bool:
    """Führt Plausibilitätschecks für Dual-Magnet-Ergebnisse durch."""
    all_pass = True

    print("\n" + "=" * 60)
    print("PLAUSIBILITAETSCHECKS (Dual-Magnet)")
    print("=" * 60)

    for res in results:
        label = res["label"]
        B = res["B"]
        B1 = res["B1"]
        B2 = res["B2"]
        dist1 = res["distances1"]
        dist2 = res["distances2"]

        # Check 1: NaN / Inf
        has_nan = np.isnan(B).any()
        has_inf = np.isinf(B).any()
        status = "FAIL" if (has_nan or has_inf) else "PASS"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {label}: NaN={has_nan}, Inf={has_inf}")

        # Check 2: Superposition korrekt (B_total ≈ B1 + B2)
        B_sum = B1 + B2
        superpos_ok = np.allclose(B, B_sum, rtol=1e-10)
        status = "PASS" if superpos_ok else "FAIL"
        if status == "FAIL":
            all_pass = False
            max_diff = np.max(np.abs(B - B_sum))
            print(f"  [{status}] {label}: Superposition FAILED, max_diff={max_diff:.3e}")
        else:
            print(f"  [{status}] {label}: Superposition B_total = B1 + B2 ✓")

        # Check 3: Feldstärken im plausiblen Bereich
        B_abs = np.abs(B)
        B_max = np.max(B_abs)
        B_nonzero = B_abs[B_abs > 0]
        B_min_nz = np.min(B_nonzero) if len(B_nonzero) > 0 else 0.0
        reasonable = 1e-10 < B_max < 10.0
        status = "PASS" if reasonable else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {label}: |B|_max={B_max:.3e} T, |B|_min_nz={B_min_nz:.3e} T")

        # Check 4: Abstände > Magnetradius (kein Sensor im Magneten)
        magnet_radius = DIAMETER / 2
        min_d1 = np.min(dist1)
        min_d2 = np.min(dist2)
        status = "PASS" if (min_d1 > magnet_radius and min_d2 > magnet_radius) else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {label}: min_d1={min_d1:.4f} m, min_d2={min_d2:.4f} m "
              f"(> {magnet_radius} m)")

        # Check 5: Pfadlänge
        expected_len = N_PATH + 1
        actual_len = len(B)
        status = "PASS" if actual_len == expected_len else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {label}: Pfadlaenge={actual_len} (erwartet {expected_len})")

        # Check 6: Magnet 2 näher als Magnet 1 (im Mittel)
        mean_d1 = np.mean(dist1)
        mean_d2 = np.mean(dist2)
        status = "PASS" if mean_d2 < mean_d1 else "WARN"
        print(f"  [{status}] {label}: mean_d1={mean_d1:.4f} m, mean_d2={mean_d2:.4f} m "
              f"(M2 soll naeher sein)")

        # Check 7: Dual-Feld stärker als stärkstes Einzelfeld?
        # (Nicht zwingend wegen Kompensation, aber informativ)
        B_norm_total = np.mean(np.linalg.norm(B, axis=1))
        B_norm_1 = np.mean(np.linalg.norm(B1, axis=1))
        B_norm_2 = np.mean(np.linalg.norm(B2, axis=1))
        print(f"  [INFO] {label}: mean|B|={B_norm_total:.4e} T, "
              f"mean|B1|={B_norm_1:.4e} T, mean|B2|={B_norm_2:.4e} T")

    # Globaler Check: Quadranten (alle B-Komponenten positiv und negativ?)
    all_B = np.concatenate([r["B"] for r in results], axis=0)
    has_pos_x = np.any(all_B[:, 0] > 0)
    has_neg_x = np.any(all_B[:, 0] < 0)
    has_pos_y = np.any(all_B[:, 1] > 0)
    has_neg_y = np.any(all_B[:, 1] < 0)
    has_pos_z = np.any(all_B[:, 2] > 0)
    has_neg_z = np.any(all_B[:, 2] < 0)
    quadrant_ok = has_pos_x and has_neg_x and has_pos_y and has_neg_y
    status = "PASS" if quadrant_ok else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"\n  [{status}] Quadranten: Bx+={has_pos_x}, Bx-={has_neg_x}, "
          f"By+={has_pos_y}, By-={has_neg_y}, Bz+={has_pos_z}, Bz-={has_neg_z}")

    print("-" * 60)
    final = "PASS" if all_pass else "FAIL"
    print(f"  GESAMTERGEBNIS: [{final}]")
    print("=" * 60)
    return all_pass


# ==============================================================
# VISUALISIERUNG
# ==============================================================

def plot_results(results: list[dict], info_gains: list[dict]):
    """Erstellt alle Plots und speichert sie als PNG."""
    n_orient = len(results)
    colors = plt.cm.hsv(np.linspace(0, 0.95, n_orient))
    path_idx = np.arange(results[0]["B"].shape[0])

    # ----------------------------------------------------------
    # Plot 1: 3D-Pfade beider Magnete (Top-10 nach Entropie)
    # ----------------------------------------------------------
    entropies_norm = [ig["entropy_normalized"] for ig in info_gains]
    ent_finite = [e if np.isfinite(e) else -1e30 for e in entropies_norm]
    top_indices = np.argsort(ent_finite)[-10:][::-1]

    fig1 = plt.figure(figsize=(12, 9))
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.scatter(*SENSOR_POS, color="red", s=200, marker="*", label="Sensor", zorder=5)
    ax1.scatter(*PIVOT1, color="black", s=100, marker="D", label="Pivot1", zorder=5)
    ax1.scatter(*PIVOT2, color="gray", s=100, marker="D", label="Pivot2", zorder=5)

    top_colors = plt.cm.tab10(np.linspace(0, 1, len(top_indices)))
    for ci, idx in enumerate(top_indices):
        res = results[idx]
        p1 = res["positions1"]
        p2 = res["positions2"]
        short = res["label"][:25]
        ax1.plot(p1[:, 0], p1[:, 1], p1[:, 2],
                 color=top_colors[ci], linewidth=1.5, label=f"{short} (M1)")
        ax1.plot(p2[:, 0], p2[:, 1], p2[:, 2],
                 color=top_colors[ci], linewidth=1.5, linestyle="--")

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("Dual-Magnet: Top-10 Pfade (nach Norm. Entropie)")
    ax1.legend(fontsize=6, loc="upper left")
    plt.tight_layout()
    plt.savefig("dual_plot_3d_pfade.png", dpi=150, bbox_inches="tight")
    print("  -> dual_plot_3d_pfade.png gespeichert")

    # ----------------------------------------------------------
    # Plot 2: |B| für Top-10
    # ----------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    for ci, idx in enumerate(top_indices):
        res = results[idx]
        B_norm = np.linalg.norm(res["B"], axis=1) * 1e3
        ax2.plot(path_idx, B_norm, color=top_colors[ci], linewidth=1.5,
                 label=res["label"][:30])
    ax2.set_xlabel("Pfadindex")
    ax2.set_ylabel("|B| (mT)")
    ax2.set_title("Dual-Magnet: |B| Top-10 Konfigurationen")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("dual_plot_b_norm.png", dpi=150, bbox_inches="tight")
    print("  -> dual_plot_b_norm.png gespeichert")

    # ----------------------------------------------------------
    # Plot 3: B-Komponenten für Top-5
    # ----------------------------------------------------------
    top5 = top_indices[:5]
    fig3, axes3 = plt.subplots(len(top5), 3, figsize=(15, 3 * len(top5)), sharex=True)
    if len(top5) == 1:
        axes3 = axes3[np.newaxis, :]
    comp_labels = ["Bx", "By", "Bz"]
    top5_colors = plt.cm.tab10(np.linspace(0, 1, len(top5)))

    for ri, idx in enumerate(top5):
        B = results[idx]["B"]
        for j in range(3):
            axes3[ri, j].plot(path_idx, B[:, j] * 1e3, color=top5_colors[ri], linewidth=1)
            axes3[ri, j].set_ylabel(f"{comp_labels[j]} (mT)")
            axes3[ri, j].grid(True, alpha=0.3)
            if ri == 0:
                axes3[ri, j].set_title(comp_labels[j])
        axes3[ri, 0].annotate(
            results[idx]["label"][:30], xy=(0.02, 0.92), xycoords="axes fraction",
            fontsize=7, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="wheat", alpha=0.5),
        )
    axes3[-1, 1].set_xlabel("Pfadindex")
    fig3.suptitle("Dual-Magnet: B-Komponenten Top-5", fontsize=14)
    plt.tight_layout()
    plt.savefig("dual_plot_b_felder.png", dpi=150, bbox_inches="tight")
    print("  -> dual_plot_b_felder.png gespeichert")

    # ----------------------------------------------------------
    # Plot 4: Erweiterte Metriken (6er-Grid)
    # ----------------------------------------------------------
    labels_short = [r["label"][:22] for r in results]
    x_pos = np.arange(n_orient)

    fig4, axes4 = plt.subplots(2, 3, figsize=(22, 12))

    # Normalisierte Entropie
    ent_plot = [e if np.isfinite(e) else np.nan for e in entropies_norm]
    best_ent = int(np.nanargmax(ent_plot))
    bar_c = ["gold" if i == best_ent else colors[i] for i in range(n_orient)]
    edge_c = ["red" if i == best_ent else "none" for i in range(n_orient)]
    lw = [3 if i == best_ent else 0 for i in range(n_orient)]
    axes4[0, 0].bar(x_pos, ent_plot, color=bar_c, edgecolor=edge_c, linewidth=lw)
    axes4[0, 0].set_xticks(x_pos)
    axes4[0, 0].set_xticklabels(labels_short, rotation=90, ha="center", fontsize=5)
    axes4[0, 0].set_ylabel("Norm. Entropie")
    axes4[0, 0].set_title(f"[EMPFOHLEN] Norm. Entropie\nBeste: {labels_short[best_ent]}", fontsize=9)
    axes4[0, 0].grid(axis="y", alpha=0.3)

    # SNR
    snrs = [ig["snr"] for ig in info_gains]
    best_snr = int(np.argmax(snrs))
    bar_c = ["gold" if i == best_snr else colors[i] for i in range(n_orient)]
    edge_c = ["red" if i == best_snr else "none" for i in range(n_orient)]
    lw = [3 if i == best_snr else 0 for i in range(n_orient)]
    axes4[0, 1].bar(x_pos, snrs, color=bar_c, edgecolor=edge_c, linewidth=lw)
    axes4[0, 1].set_xticks(x_pos)
    axes4[0, 1].set_xticklabels(labels_short, rotation=90, ha="center", fontsize=5)
    axes4[0, 1].set_ylabel("SNR")
    axes4[0, 1].set_title(f"SNR\nBeste: {labels_short[best_snr]}", fontsize=9)
    axes4[0, 1].grid(axis="y", alpha=0.3)

    # Normalisierte Varianz
    norm_vars = [ig["normalized_var"] for ig in info_gains]
    best_nv = int(np.argmax(norm_vars))
    bar_c = ["gold" if i == best_nv else colors[i] for i in range(n_orient)]
    edge_c = ["red" if i == best_nv else "none" for i in range(n_orient)]
    lw = [3 if i == best_nv else 0 for i in range(n_orient)]
    axes4[0, 2].bar(x_pos, norm_vars, color=bar_c, edgecolor=edge_c, linewidth=lw)
    axes4[0, 2].set_xticks(x_pos)
    axes4[0, 2].set_xticklabels(labels_short, rotation=90, ha="center", fontsize=5)
    axes4[0, 2].set_ylabel("Norm. Varianz")
    axes4[0, 2].set_title(f"Norm. Varianz\nBeste: {labels_short[best_nv]}", fontsize=9)
    axes4[0, 2].grid(axis="y", alpha=0.3)

    # Effektive Dimensionalität
    eff_dims = [ig["effective_dim"] for ig in info_gains]
    best_ed = int(np.argmax(eff_dims))
    bar_c = ["gold" if i == best_ed else colors[i] for i in range(n_orient)]
    edge_c = ["red" if i == best_ed else "none" for i in range(n_orient)]
    lw = [3 if i == best_ed else 0 for i in range(n_orient)]
    axes4[1, 0].bar(x_pos, eff_dims, color=bar_c, edgecolor=edge_c, linewidth=lw)
    axes4[1, 0].set_xticks(x_pos)
    axes4[1, 0].set_xticklabels(labels_short, rotation=90, ha="center", fontsize=5)
    axes4[1, 0].set_ylabel("Eff. Dimensionalität")
    axes4[1, 0].set_ylim([0, 3.5])
    axes4[1, 0].axhline(y=3.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes4[1, 0].set_title(f"Eff. Dim. (3=isotrop)\nBeste: {labels_short[best_ed]}", fontsize=9)
    axes4[1, 0].grid(axis="y", alpha=0.3)

    # Volumetrische Effizienz
    vol_effs = [ig["volumetric_efficiency"] for ig in info_gains]
    best_ve = int(np.argmax(vol_effs))
    bar_c = ["gold" if i == best_ve else colors[i] for i in range(n_orient)]
    edge_c = ["red" if i == best_ve else "none" for i in range(n_orient)]
    lw = [3 if i == best_ve else 0 for i in range(n_orient)]
    axes4[1, 1].bar(x_pos, vol_effs, color=bar_c, edgecolor=edge_c, linewidth=lw)
    axes4[1, 1].set_xticks(x_pos)
    axes4[1, 1].set_xticklabels(labels_short, rotation=90, ha="center", fontsize=5)
    axes4[1, 1].set_ylabel("Vol. Effizienz")
    axes4[1, 1].set_ylim([0, 1.1])
    axes4[1, 1].axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes4[1, 1].set_title(f"Vol. Effizienz (1=isotrop)\nBeste: {labels_short[best_ve]}", fontsize=9)
    axes4[1, 1].grid(axis="y", alpha=0.3)

    # Konditionszahl
    cond_nums = [ig["condition_num"] for ig in info_gains]
    cond_plot = [c if np.isfinite(c) else np.nan for c in cond_nums]
    finite_conds = [(c, i) for i, c in enumerate(cond_plot) if np.isfinite(c)]
    best_cond = min(finite_conds, key=lambda x: x[0])[1] if finite_conds else 0
    bar_c = ["gold" if i == best_cond else colors[i] for i in range(n_orient)]
    edge_c = ["red" if i == best_cond else "none" for i in range(n_orient)]
    lw = [3 if i == best_cond else 0 for i in range(n_orient)]
    axes4[1, 2].bar(x_pos, cond_plot, color=bar_c, edgecolor=edge_c, linewidth=lw)
    axes4[1, 2].set_xticks(x_pos)
    axes4[1, 2].set_xticklabels(labels_short, rotation=90, ha="center", fontsize=5)
    axes4[1, 2].set_ylabel("Konditionszahl")
    axes4[1, 2].set_yscale("log")
    axes4[1, 2].set_title(f"Konditionszahl (niedrig=besser)\nBeste: {labels_short[best_cond]}", fontsize=9)
    axes4[1, 2].grid(axis="y", alpha=0.3)

    fig4.suptitle("Dual-Magnet: Informationsgewinn-Metriken", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("dual_plot_informationsgewinn.png", dpi=150, bbox_inches="tight")
    print("  -> dual_plot_informationsgewinn.png gespeichert")

    # ----------------------------------------------------------
    # Plot 5: Ranking-Tabelle (Top-15)
    # ----------------------------------------------------------
    ranking_idx = np.argsort(ent_finite)[::-1]
    top_n = min(15, n_orient)

    fig5, ax5 = plt.subplots(figsize=(20, max(6, top_n * 0.4)))
    ax5.axis("off")

    table_data = []
    for rank, idx in enumerate(ranking_idx[:top_n]):
        ent_str = f"{entropies_norm[idx]:.4f}" if np.isfinite(entropies_norm[idx]) else "-inf"
        table_data.append([
            f"{rank+1}",
            results[idx]["label"],
            ent_str,
            f"{snrs[idx]:.4f}",
            f"{norm_vars[idx]:.4f}",
            f"{eff_dims[idx]:.3f}",
            f"{vol_effs[idx]:.4f}",
            f"{cond_nums[idx]:.1f}" if np.isfinite(cond_nums[idx]) else "inf",
        ])

    col_labels = ["Rang", "Konfiguration", "Entropie\n(norm.)", "SNR",
                  "Norm.\nVarianz", "Eff.\nDim.", "Vol.\nEff.", "Kond.\nZahl"]
    table = ax5.table(cellText=table_data, colLabels=col_labels,
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for rank in range(min(3, top_n)):
        for j in range(len(col_labels)):
            table[rank + 1, j].set_facecolor("#A9DFBF" if rank == 0 else "#D4E6F1")

    ax5.set_title("Dual-Magnet: Top-15 Ranking (Normalisierte Entropie)",
                   fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("dual_plot_ranking.png", dpi=150, bbox_inches="tight")
    print("  -> dual_plot_ranking.png gespeichert")

    # ----------------------------------------------------------
    # Plot 6: Vergleich Einzel- vs. Gesamt-Feld (Top-5)
    # ----------------------------------------------------------
    fig6, axes6 = plt.subplots(len(top5), 1, figsize=(14, 3 * len(top5)), sharex=True)
    if len(top5) == 1:
        axes6 = [axes6]

    for ri, idx in enumerate(top5):
        res = results[idx]
        B_norm = np.linalg.norm(res["B"], axis=1) * 1e3
        B1_norm = np.linalg.norm(res["B1"], axis=1) * 1e3
        B2_norm = np.linalg.norm(res["B2"], axis=1) * 1e3
        axes6[ri].plot(path_idx, B_norm, 'k-', linewidth=2, label="|B_total|")
        axes6[ri].plot(path_idx, B1_norm, 'b--', linewidth=1, label="|B1| (M1)")
        axes6[ri].plot(path_idx, B2_norm, 'r--', linewidth=1, label="|B2| (M2)")
        axes6[ri].set_ylabel("|B| (mT)")
        axes6[ri].grid(True, alpha=0.3)
        axes6[ri].legend(fontsize=7, loc="upper right")
        axes6[ri].set_title(res["label"][:40], fontsize=9)

    axes6[-1].set_xlabel("Pfadindex")
    fig6.suptitle("Dual-Magnet: Einzel- vs. Gesamtfeld (Top-5)", fontsize=14)
    plt.tight_layout()
    plt.savefig("dual_plot_einzel_vs_gesamt.png", dpi=150, bbox_inches="tight")
    print("  -> dual_plot_einzel_vs_gesamt.png gespeichert")

    # ----------------------------------------------------------
    # Plot 7: Abstände
    # ----------------------------------------------------------
    fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(14, 5))
    for ci, idx in enumerate(top_indices):
        res = results[idx]
        ax7a.plot(path_idx, res["distances1"] * 100, color=top_colors[ci],
                  linewidth=1, label=res["label"][:22])
        ax7b.plot(path_idx, res["distances2"] * 100, color=top_colors[ci],
                  linewidth=1, label=res["label"][:22])
    ax7a.set_xlabel("Pfadindex")
    ax7a.set_ylabel("Abstand (cm)")
    ax7a.set_title("Abstand Sensor ↔ Magnet 1")
    ax7a.legend(fontsize=5)
    ax7a.grid(True, alpha=0.3)
    ax7b.set_xlabel("Pfadindex")
    ax7b.set_ylabel("Abstand (cm)")
    ax7b.set_title("Abstand Sensor ↔ Magnet 2")
    ax7b.legend(fontsize=5)
    ax7b.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("dual_plot_abstaende.png", dpi=150, bbox_inches="tight")
    print("  -> dual_plot_abstaende.png gespeichert")

    plt.close("all")


# ==============================================================
# ANIMATIONEN (Top-5)
# ==============================================================

def create_animations(results: list[dict], info_gains: list[dict]):
    """Erstellt interaktive Plotly-Animationen für die Top-5 Konfigurationen."""
    print("\nErstelle Animationen (Top-5, Plotly Backend) ...")

    entropies_norm = [ig["entropy_normalized"] for ig in info_gains]
    ent_finite = [e if np.isfinite(e) else -1e30 for e in entropies_norm]
    top5 = np.argsort(ent_finite)[-5:][::-1]

    for rank, idx in enumerate(top5):
        res = results[idx]
        label = res["label"]
        coll = res["collection"]

        sensor = magpy.Sensor(position=SENSOR_POS)

        safe_label = (
            label.replace("/", "-").replace("||", "_").replace("°", "deg")
            .replace("|", "_").replace(":", "_").replace("(", "")
            .replace(")", "").replace(" ", "_").replace(">", "")
            .replace("<", "").replace("-", "")
        )
        filename = f"dual_animation_{rank+1:02d}_{safe_label[:40]}.html"

        print(f"  [Rang {rank+1}] {label} ... ", end="", flush=True)

        try:
            fig = magpy.show(
                coll,
                sensor,
                animation=3,
                animation_slider=True,
                animation_fps=20,
                backend='plotly',
                return_fig=True,
                style_path_show=True,
                style_legend_show=True,
            )

            fig.update_layout(
                title=dict(
                    text=f"Dual-Magnet Rang {rank+1}: {label}",
                    x=0.5, xanchor='center',
                    font=dict(size=14, color='darkblue'),
                ),
                scene=dict(
                    xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
                    aspectmode='data',
                ),
                width=1200, height=800,
            )

            fig.write_html(filename, config={
                'displayModeBar': True, 'displaylogo': False,
            })
            print(f"-> {filename}")

        except Exception as e:
            print(f"FEHLER: {e}")
            continue

    print("Animationen erstellt.")


# ==============================================================
# HAUPTPROGRAMM
# ==============================================================

def main():
    print("=" * 70)
    print("  Dual-Magnet Magnetfeld-Simulation")
    print("=" * 70)
    print(f"  Magnet:     Zylinder D={DIAMETER*1e3:.0f} mm, H={HEIGHT*1e3:.0f} mm")
    print(f"  Magnetis.:  M = {MAG_VECTOR} A/m  (J ~ {4*np.pi*1e-7*MAG_VECTOR[2]:.3f} T)")
    print(f"  Sensor:     {SENSOR_POS} m")
    print(f"  Magnet1:    {MAGNET1_INIT_POS} m  (Pivot: {PIVOT1} m)")
    print(f"  Magnet2:    {MAGNET2_INIT_POS} m  (Pivot: {PIVOT2} m)")
    print(f"  Abstand M1: {np.linalg.norm(MAGNET1_INIT_POS - SENSOR_POS)*100:.1f} cm")
    print(f"  Abstand M2: {np.linalg.norm(MAGNET2_INIT_POS - SENSOR_POS)*100:.1f} cm  (naeher!)")
    print(f"  Schwingung: +/-{SWING_ANGLE} deg, {N_PATH} Pfadpunkte")
    print(f"  M1-Orient.: {len(M1_ORIENTATIONS)} Varianten")
    print(f"  M2-Orient.: {len(M2_ORIENTATIONS)} Varianten")
    print(f"  Kombi:      {len(DUAL_ORIENTATIONS)} Konfigurationen")
    print()

    angles = generate_pendulum_angles(N_HALF, SWING_ANGLE)
    print(f"  Pfadwinkel: {len(angles)} Punkte  [{angles.min():.1f} .. {angles.max():.1f}] deg")

    # ==========================================================
    # Simulation für alle Orientierungs-Kombinationen
    # ==========================================================
    results: list[dict] = []
    info_gains: list[dict] = []

    for label, m1_rot, m2_rot in DUAL_ORIENTATIONS:
        print(f"\n  Simuliere: {label} ... ", end="", flush=True)
        res = simulate_dual_orientation(label, m1_rot, m2_rot, angles)
        ig = compute_info_gain(res["B"])
        results.append(res)
        info_gains.append(ig)

        B_norms = np.linalg.norm(res["B"], axis=1)
        print(f"|B| = [{B_norms.min():.3e}, {B_norms.max():.3e}] T,  "
              f"H_norm = {ig['entropy_normalized']:.4f}" if np.isfinite(ig['entropy_normalized'])
              else f"|B| = [{B_norms.min():.3e}, {B_norms.max():.3e}] T,  H_norm = -inf")

    # ==========================================================
    # Plausibilitätschecks
    # ==========================================================
    plausibility_checks(results)

    # ==========================================================
    # Beste Konfiguration
    # ==========================================================
    entropies_norm = [ig["entropy_normalized"] for ig in info_gains]
    ent_finite = [e if np.isfinite(e) else -1e30 for e in entropies_norm]
    snrs = [ig["snr"] for ig in info_gains]
    eff_dims = [ig["effective_dim"] for ig in info_gains]
    vol_effs = [ig["volumetric_efficiency"] for ig in info_gains]
    total_vars = [ig["total_var"] for ig in info_gains]

    best_ent = int(np.argmax(ent_finite))
    best_snr = int(np.argmax(snrs))
    best_ed = int(np.argmax(eff_dims))
    best_ve = int(np.argmax(vol_effs))
    best_var = int(np.argmax(total_vars))

    print(f"\n{'='*70}")
    print(f"  ERGEBNISSE - DUAL-MAGNET INFORMATIONSGEWINN")
    print(f"{'='*70}")

    print(f"\n  [EMPFOHLEN] Normalisierte Entropie:")
    print(f"    Konfiguration:     {results[best_ent]['label']}")
    print(f"    Entropie (norm.):  {entropies_norm[best_ent]:.4f}")
    print(f"    Eff. Dimension:    {eff_dims[best_ent]:.4f} / 3.0")
    print(f"    Vol. Effizienz:    {vol_effs[best_ent]:.4f}")
    print(f"    SNR:               {snrs[best_ent]:.4f}")

    print(f"\n  Alternative Kriterien:")
    print(f"    Max. SNR:          {results[best_snr]['label']}  ({snrs[best_snr]:.4f})")
    print(f"    Max. Eff. Dim.:    {results[best_ed]['label']}  ({eff_dims[best_ed]:.4f})")
    print(f"    Max. Vol. Eff.:    {results[best_ve]['label']}  ({vol_effs[best_ve]:.4f})")
    print(f"    Max. Total Var.:   {results[best_var]['label']}  ({total_vars[best_var]:.3e})")

    # Top-10 Ranking
    ranking = np.argsort(ent_finite)[::-1]
    print(f"\n  {'─'*70}")
    print(f"  Top-10 Ranking (Normalisierte Entropie)")
    print(f"  {'─'*70}")
    print(f"  {'Rang':<5} {'Konfiguration':<35} {'H_norm':>8} {'Eff.D.':>7} {'SNR':>8}")
    print(f"  {'─'*70}")
    for rank, idx in enumerate(ranking[:10]):
        ent_str = f"{entropies_norm[idx]:.4f}" if np.isfinite(entropies_norm[idx]) else "-inf"
        print(f"  {rank+1:<5} {results[idx]['label']:<35} {ent_str:>8} "
              f"{eff_dims[idx]:>7.3f} {snrs[idx]:>8.4f}")
    print(f"  {'─'*70}")
    print(f"{'='*70}")

    # ==========================================================
    # Plots
    # ==========================================================
    print("\nErstelle Plots ...")
    plot_results(results, info_gains)
    print("Alle Plots gespeichert.")

    # ==========================================================
    # Animationen (nur Top-5)
    # ==========================================================
    create_animations(results, info_gains)

    print("\nDual-Magnet-Simulation abgeschlossen.")


if __name__ == "__main__":
    main()
