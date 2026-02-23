"""
Magnetfeld-Simulation mit Magpylib
====================================
Simuliert einen Zylindermagneten, der pendelnd um einen Drehpunkt schwingt,
und berechnet das B-Feld an einem fixen Sensor für verschiedene
Magnetausrichtungen. Bestimmt die Ausrichtung mit maximalem
Informationsgewinn (Varianz der Feldstärke).

Einheiten: SI (Meter, Tesla, A/m)
Autor: Magpy-Simulation
"""

import numpy as np
import magpylib as magpy
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use("Agg")  # Non-interactive Backend für zuverlässiges Speichern
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ==============================================================
# KONFIGURATION
# ==============================================================

# Magnet-Parameter (Zylinder)
DIAMETER = 0.01          # 1 cm Durchmesser
HEIGHT = 0.01            # 1 cm Höhe
MAG_VECTOR = (0, 0, 1e6)  # Magnetisierung in A/m (J ≈ µ0*M ≈ 1.26 T)

# Geometrie (SI: Meter)
SENSOR_POS = np.array([0.0, 0.0, 0.0])
MAGNET_INIT_POS = np.array([0.04, 0.0, 0.0])   # 4 cm in x-Richtung
PIVOT = np.array([0.02, 0.0, 0.0])              # Drehpunkt bei 2 cm

# Pfad-Parameter
SWING_ANGLE = 30   # ±30 Grad Schwingung um z-Achse
N_HALF = 50        # Punkte pro Halbschwingung
N_PATH = 2 * N_HALF  # 100 Punkte gesamt

# Magnetausrichtungen (Label, Rotation oder None für Default)
ORIENTATIONS = [
    # Hauptachsen
    ("M || +z (Standard)",       None),
    ("M || +x (90° um y)",       R.from_euler("y", 90, degrees=True)),
    ("M || +y (90° um x)",       R.from_euler("x", 90, degrees=True)),
    ("M || -z (180° um y)",      R.from_euler("y", 180, degrees=True)),
    ("M || -x (90° um y)",       R.from_euler("y", -90, degrees=True)),
    ("M || -y (-90° um x)",      R.from_euler("x", -90, degrees=True)),
    
    # 30° Rotationen um einzelne Achsen
    ("M 30° um x",               R.from_euler("x", 30, degrees=True)),
    ("M 30° um y",               R.from_euler("y", 30, degrees=True)),
    ("M 30° um z",               R.from_euler("z", 30, degrees=True)),
    ("M -30° um x",              R.from_euler("x", -30, degrees=True)),
    ("M -30° um y",              R.from_euler("y", -30, degrees=True)),
    
    # 60° Rotationen um einzelne Achsen
    ("M 60° um x",               R.from_euler("x", 60, degrees=True)),
    ("M 60° um y",               R.from_euler("y", 60, degrees=True)),
    ("M 60° um z",               R.from_euler("z", 60, degrees=True)),
    ("M -60° um x",              R.from_euler("x", -60, degrees=True)),
    ("M -60° um y",              R.from_euler("y", -60, degrees=True)),
    
    # 45° Rotationen (ursprünglich vorhanden)
    ("M 45° um x",               R.from_euler("x", 45, degrees=True)),
    ("M 45° um y",               R.from_euler("y", 45, degrees=True)),
    ("M 45° um z",               R.from_euler("z", 45, degrees=True)),
    
    # Kombinierte Rotationen 30°
    ("M 30°+30° um xy",          R.from_euler("xy", [30, 30], degrees=True)),
    ("M 30°+30° um xz",          R.from_euler("xz", [30, 30], degrees=True)),
    ("M 30°+30° um yz",          R.from_euler("yz", [30, 30], degrees=True)),
    
    # Kombinierte Rotationen 60°
    ("M 60°+60° um xy",          R.from_euler("xy", [60, 60], degrees=True)),
    ("M 60°+60° um xz",          R.from_euler("xz", [60, 60], degrees=True)),
    ("M 60°+60° um yz",          R.from_euler("yz", [60, 60], degrees=True)),
    
    # Kombinierte Rotationen 45° (ursprünglich)
    ("M 45°+45° um xy",          R.from_euler("xy", [45, 45], degrees=True)),
    ("M 45°+45° um xz",          R.from_euler("xz", [45, 45], degrees=True)),
    ("M 45°+45° um yz",          R.from_euler("yz", [45, 45], degrees=True)),
    
    # Gemischte Winkel
    ("M 30°+60° um xy",          R.from_euler("xy", [30, 60], degrees=True)),
    ("M 60°+30° um yz",          R.from_euler("yz", [60, 30], degrees=True)),
    ("M 30°+45° um xz",          R.from_euler("xz", [30, 45], degrees=True)),
]


# ==============================================================
# HILFSFUNKTIONEN
# ==============================================================

def generate_pendulum_angles(n_half: int, swing_deg: float) -> np.ndarray:
    """Erzeugt Pendelwinkel: hin (-swing -> +swing) und zurück (+swing -> -swing)."""
    fwd = np.linspace(-swing_deg, swing_deg, n_half)
    bwd = np.linspace(swing_deg, -swing_deg, n_half)
    return np.concatenate([fwd, bwd])


def simulate_single_orientation(
    orient_label: str,
    orient_rotation: R | None,
    angles: np.ndarray,
) -> dict:
    """
    Simuliert den Magneten für eine bestimmte Ausrichtung über den gesamten Pendelpfad.

    Args:
        orient_label:    Beschreibung der Orientierung
        orient_rotation: Scipy Rotation oder None (Default = M entlang z)
        angles:          Array der Pendelwinkel in Grad

    Returns:
        dict mit keys: 'label', 'B', 'positions', 'distances', 'magnet'
    """
    # Sensor fix am Ursprung
    sensor = magpy.Sensor(position=SENSOR_POS)

    # Magnet erstellen (Magnetisierung in A/m -> Magpylib rechnet intern in T)
    magnet = magpy.magnet.Cylinder(
        dimension=(DIAMETER, HEIGHT),
        magnetization=MAG_VECTOR,
        position=MAGNET_INIT_POS.copy(),
    )

    # Physische Ausrichtung setzen (Rotation um eigenen Mittelpunkt)
    # anchor=None -> Rotation um magnet.position (ändert nur Orientierung, nicht Position)
    if orient_rotation is not None:
        magnet.rotate(orient_rotation, anchor=None)

    # Pendelpfad anhängen: jeder Winkel erzeugt einen neuen Pfadeintrag
    # Rotation um z-Achse mit Drehpunkt PIVOT
    magnet.rotate_from_angax(angle=angles, axis="z", anchor=PIVOT)

    # B-Feld berechnen (vektorisiert über gesamten Pfad)
    B = magpy.getB(magnet, sensor)  # shape: (1 + N_PATH, 3)

    # Positionen und Abstände extrahieren
    positions = magnet.position     # shape: (1 + N_PATH, 3)
    distances = np.linalg.norm(positions - SENSOR_POS, axis=1)

    return {
        "label": orient_label,
        "B": B,
        "positions": positions,
        "distances": distances,
        "magnet": magnet,
    }


def compute_info_gain(B: np.ndarray) -> dict:
    """
    Berechnet den Informationsgewinn über mehrere Metriken.

    PRIMÄRE METRIKEN (dimensionslos, fair vergleichbar):
    =====================================================
    1. **Normalisierte Entropie (entropy_normalized):** [EMPFOHLEN]
       Differentielle Entropie der normalisierten Kovarianzmatrix.
       Skalierungsinvariant, informationstheoretisch fundiert.
       Höher = mehr Information über verschiedene Zustände.

    2. **SNR (Signal-to-Noise Ratio):** Std(|B|) / Mean(|B|)
       Variationskoeffizient. Höher = stärkere relative Variation.
       Gut für: Trajektorien-Unterscheidung.

    3. **Normalisierte Varianz (normalized_var):** total_var / Mean(|B|)²
       Dimensionslose Gesamtvarianz aller 3 Komponenten.
       Äquivalent zu SNR² für 1D, berücksichtigt aber alle Achsen.

    4. **Effektive Dimensionalität (effective_dim):** 1.0 bis 3.0
       Shannon-basiert: Wie viele Achsen tragen zur Variation bei?
       3.0 = alle Achsen gleichmäßig, 1.0 = nur eine Achse variiert.
       Gut für: Sensor-Kalibrierung (nahe 3 anstreben).

    5. **Volumetrische Effizienz (volumetric_efficiency):** 0.0 bis 1.0
       Verhältnis geometr. zu arithm. Mittel der Eigenwerte.
       1.0 = perfekt isotrop, <1 = anisotrop.
       Gut für: gleichmäßige Achsennutzung.

    SEKUNDÄRE METRIKEN (absolut, einheitenabhängig):
    =================================================
    - total_var: Spur(Cov) in T² – ignoriert Korrelationen
    - det_cov: Determinante in T⁶ – Volumen der Info-Ellipse
    - entropy: Absolute Entropie – einheitenabhängig, schwer interpretierbar
    - condition_num: λ_max/λ_min – Anisotropie (niedriger = besser)

    Returns:
        dict mit 19 Metriken zur umfassenden Charakterisierung
    """
    B_norm = np.linalg.norm(B, axis=1)
    mean_norm = np.mean(B_norm)

    # Basis-Statistiken (ddof=1 für Konsistenz mit np.cov)
    var_norm = np.var(B_norm, ddof=1)
    std_norm = np.std(B_norm, ddof=1)
    var_components = np.var(B, axis=0, ddof=1)  # [var_Bx, var_By, var_Bz]
    std_components = np.std(B, axis=0, ddof=1)
    total_var = np.sum(var_components)           # Spur der Kovarianzmatrix

    # Vollständige Kovarianzmatrix (3x3) – erfasst auch Korrelationen
    cov_matrix = np.cov(B.T)                    # shape: (3, 3), ddof=1

    # Eigenwerte der Kovarianzmatrix
    eigenvalues = np.linalg.eigvalsh(cov_matrix)  # sortiert aufsteigend
    eigenvalues = np.maximum(eigenvalues, 0.0)     # numerische Stabilität

    # Determinante = Produkt der Eigenwerte = Volumen der Info-Ellipse
    det_cov = np.prod(eigenvalues)

    # Differentielle Entropie einer multivariaten Normalverteilung
    # h(X) = 0.5 * ln( (2*pi*e)^d * det(Cov) )
    d = 3  # Dimensionen
    if det_cov > 0:
        entropy = 0.5 * (d * np.log(2 * np.pi * np.e) + np.log(det_cov))
    else:
        entropy = -np.inf  # degeneriert (eine Achse hat 0 Varianz)

    # NORMALISIERTE Entropie (dimensionslos, skalierungsinvariant)
    # Dividiere durch mean_norm^2 pro Dimension → dimensionslos
    if mean_norm > 1e-20 and det_cov > 0:
        # Kovarianzmatrix normalisieren
        cov_normalized = cov_matrix / (mean_norm ** 2)
        det_cov_norm = np.linalg.det(cov_normalized)
        if det_cov_norm > 0:
            entropy_normalized = 0.5 * (d * np.log(2 * np.pi * np.e) + np.log(det_cov_norm))
        else:
            entropy_normalized = -np.inf
    else:
        entropy_normalized = -np.inf

    # SNR = Variationskoeffizient des Betrags (höher = mehr relative Variation)
    snr = std_norm / mean_norm if mean_norm > 1e-20 else 0.0

    # Normalisierte Gesamtvarianz (dimensionslos)
    normalized_var = total_var / (mean_norm**2) if mean_norm > 1e-20 else 0.0

    # Konditionszahl: λ_max / λ_min (1 = isotrop, >>1 = anisotrop)
    lambda_min = eigenvalues[0]
    lambda_max = eigenvalues[-1]
    condition_num = lambda_max / lambda_min if lambda_min > 1e-30 else np.inf

    # ZUSÄTZLICHE METRIKEN für bessere Interpretation:

    # Kleinster singulärer Wert (= sqrt(λ_min)): Maß für "schwächste" Variation
    min_singular_value = np.sqrt(lambda_min)

    # Effektive Dimensionalität (wie viele Achsen tragen zur Variation bei?)
    # Basiert auf Shannon-Zahl: exp(H) / max(exp(H)) normalisiert
    if np.all(eigenvalues > 1e-30):
        # Normalisierte Eigenwerte (Wahrscheinlichkeiten)
        eig_sum = np.sum(eigenvalues)
        eig_prob = eigenvalues / eig_sum
        # Shannon-Entropie der Eigenwert-Verteilung
        shannon_eig = -np.sum(eig_prob * np.log(eig_prob + 1e-30))
        # Effektive Dimensionalität: 1 (alle Varianz in 1 Achse) bis 3 (isotrop)
        effective_dim = np.exp(shannon_eig)
    else:
        effective_dim = 0.0

    # Volumetrische Effizienz: (det(Cov))^(1/3) / mean(eigenvalues)
    # = 1 wenn isotrop, < 1 wenn anisotrop
    geom_mean_eig = det_cov ** (1/3) if det_cov > 0 else 0.0
    arith_mean_eig = np.mean(eigenvalues)
    volumetric_efficiency = geom_mean_eig / arith_mean_eig if arith_mean_eig > 1e-30 else 0.0

    return {
        # Basis
        "var_norm": var_norm,
        "std_norm": std_norm,
        "var_components": var_components,
        "std_components": std_components,
        "total_var": total_var,
        # Kovarianzstruktur
        "cov_matrix": cov_matrix,
        "eigenvalues": eigenvalues,
        "det_cov": det_cov,
        # Entropie-basiert
        "entropy": entropy,                          # Absolute Entropie (einheitenabhängig)
        "entropy_normalized": entropy_normalized,    # Normalisierte Entropie (dimensionslos)
        # Relative Metriken (dimensionslos, fair vergleichbar)
        "snr": snr,
        "normalized_var": normalized_var,
        "condition_num": condition_num,
        # Erweiterte Charakterisierung
        "min_singular_value": min_singular_value,   # Schwächste Variation
        "effective_dim": effective_dim,             # Wie viele Achsen tragen bei? (1-3)
        "volumetric_efficiency": volumetric_efficiency,  # Isotropy-Maß (0-1)
    }


# ==============================================================
# PLAUSIBILITÄTSCHECKS
# ==============================================================

def plausibility_checks(results: list[dict]) -> bool:
    """Führt Plausibilitätschecks durch und gibt PASS/FAIL aus."""
    all_pass = True

    print("\n" + "=" * 60)
    print("PLAUSIBILITAETSCHECKS")
    print("=" * 60)

    for res in results:
        label = res["label"]
        B = res["B"]
        distances = res["distances"]

        # Check 1: NaN / Inf
        has_nan = np.isnan(B).any()
        has_inf = np.isinf(B).any()
        status = "FAIL" if (has_nan or has_inf) else "PASS"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {label}: NaN={has_nan}, Inf={has_inf}")

        # Check 2: Feldstärken im plausiblen Bereich (1e-10 .. 10 T)
        B_abs = np.abs(B)
        B_max = np.max(B_abs)
        B_nonzero = B_abs[B_abs > 0]
        B_min_nz = np.min(B_nonzero) if len(B_nonzero) > 0 else 0.0
        reasonable = 1e-10 < B_max < 10.0
        status = "PASS" if reasonable else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {label}: |B|_max={B_max:.3e} T, |B|_min_nz={B_min_nz:.3e} T")

        # Check 3: Abstände > 0 und > Magnetradius (kein Inside-Magnet)
        min_dist = np.min(distances)
        magnet_radius = DIAMETER / 2
        status = "PASS" if min_dist > magnet_radius else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {label}: min_dist={min_dist:.4f} m (> {magnet_radius} m Magnet-R)")

        # Check 4: Pfadlänge korrekt
        expected_len = N_PATH + 1  # Basisposition + N_PATH appended
        actual_len = len(B)
        status = "PASS" if actual_len == expected_len else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {label}: Pfadlaenge={actual_len} (erwartet {expected_len})")

    # Check 5: Alle Quadranten (Bx, By positiv und negativ vertreten?)
    all_B = np.concatenate([r["B"] for r in results], axis=0)
    has_pos_x = np.any(all_B[:, 0] > 0)
    has_neg_x = np.any(all_B[:, 0] < 0)
    has_pos_y = np.any(all_B[:, 1] > 0)
    has_neg_y = np.any(all_B[:, 1] < 0)
    quadrant_ok = has_pos_x and has_neg_x and has_pos_y and has_neg_y
    status = "PASS" if quadrant_ok else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"\n  [{status}] Quadranten-Check: Bx+={has_pos_x}, Bx-={has_neg_x}, "
          f"By+={has_pos_y}, By-={has_neg_y}")

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
    # Verwende eine Colormap mit vielen Farben (hsv oder rainbow für mehr als 10)
    if n_orient <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_orient))
    else:
        colors = plt.cm.hsv(np.linspace(0, 0.95, n_orient))  # 0.95 um Rot-Wiederholung zu vermeiden
    path_idx = np.arange(results[0]["B"].shape[0])

    # ----------------------------------------------------------
    # Plot 1: 3D-Pfade aller Orientierungen
    # ----------------------------------------------------------
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.scatter(*SENSOR_POS, color="red", s=200, marker="*", label="Sensor", zorder=5)
    ax1.scatter(*PIVOT, color="black", s=100, marker="D", label="Drehpunkt", zorder=5)

    for i, res in enumerate(results):
        pos = res["positions"]
        ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                 color=colors[i], linewidth=1.5, label=res["label"])
        ax1.scatter(pos[0, 0], pos[0, 1], pos[0, 2],
                    color=colors[i], s=50, marker="o")

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("Magnetpfade fuer verschiedene Ausrichtungen")
    ax1.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig("plot_3d_pfade.png", dpi=150, bbox_inches="tight")
    print("  -> plot_3d_pfade.png gespeichert")

    # ----------------------------------------------------------
    # Plot 2: B-Feldkomponenten pro Ausrichtung (Subplots)
    # ----------------------------------------------------------
    fig2, axes2 = plt.subplots(n_orient, 3, figsize=(15, 3 * n_orient), sharex=True)
    if n_orient == 1:
        axes2 = axes2[np.newaxis, :]
    comp_labels = ["Bx", "By", "Bz"]

    for i, res in enumerate(results):
        B = res["B"]
        for j in range(3):
            axes2[i, j].plot(path_idx, B[:, j] * 1e3, color=colors[i], linewidth=1)
            axes2[i, j].set_ylabel(f"{comp_labels[j]} (mT)")
            axes2[i, j].grid(True, alpha=0.3)
            if i == 0:
                axes2[i, j].set_title(comp_labels[j])
        axes2[i, 0].annotate(
            res["label"], xy=(0.02, 0.92), xycoords="axes fraction",
            fontsize=8, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="wheat", alpha=0.5),
        )

    axes2[-1, 1].set_xlabel("Pfadindex")
    fig2.suptitle("B-Feldkomponenten entlang des Pfades", fontsize=14)
    plt.tight_layout()
    plt.savefig("plot_b_felder.png", dpi=150, bbox_inches="tight")
    print("  -> plot_b_felder.png gespeichert")

    # ----------------------------------------------------------
    # Plot 3: |B| (Betrag) pro Ausrichtung
    # ----------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    for i, res in enumerate(results):
        B_norm = np.linalg.norm(res["B"], axis=1) * 1e3  # mT
        ax3.plot(path_idx, B_norm, color=colors[i], linewidth=1.5, label=res["label"])
    ax3.set_xlabel("Pfadindex")
    ax3.set_ylabel("|B| (mT)")
    ax3.set_title("B-Feld Betrag entlang des Pfades")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_b_norm.png", dpi=150, bbox_inches="tight")
    print("  -> plot_b_norm.png gespeichert")

    # ----------------------------------------------------------
    # Plot 4: Informationsgewinn – Balkendiagramm (erweitert)
    # ----------------------------------------------------------
    labels_short = [res["label"] for res in results]
    x_pos = np.arange(n_orient)

    # --- 4a: Gesamtvarianz + Varianz pro Komponente (wie bisher) ---
    fig4, axes4 = plt.subplots(1, 2, figsize=(15, 6))

    total_vars = [ig["total_var"] for ig in info_gains]
    best_idx_var = int(np.argmax(total_vars))
    bar_colors = ["gold" if i == best_idx_var else colors[i] for i in range(n_orient)]
    edge_colors = ["red" if i == best_idx_var else "none" for i in range(n_orient)]
    linewidths = [3 if i == best_idx_var else 0 for i in range(n_orient)]

    bars = axes4[0].bar(x_pos, total_vars, color=bar_colors,
                         edgecolor=edge_colors, linewidth=linewidths)
    axes4[0].set_xticks(x_pos)
    axes4[0].set_xticklabels(labels_short, rotation=45, ha="right", fontsize=7)
    axes4[0].set_ylabel("Gesamtvarianz (T²)")
    axes4[0].set_title("Informationsgewinn: Gesamtvarianz (Spur Kovarianzmatrix)")
    axes4[0].bar_label(bars, fmt="%.2e", fontsize=7, padding=2)
    axes4[0].grid(axis="y", alpha=0.3)

    var_bx = np.array([ig["var_components"][0] for ig in info_gains])
    var_by = np.array([ig["var_components"][1] for ig in info_gains])
    var_bz = np.array([ig["var_components"][2] for ig in info_gains])

    axes4[1].bar(x_pos, var_bx, label="Var(Bx)", color="tab:red", alpha=0.7)
    axes4[1].bar(x_pos, var_by, bottom=var_bx, label="Var(By)", color="tab:green", alpha=0.7)
    axes4[1].bar(x_pos, var_bz, bottom=var_bx + var_by, label="Var(Bz)", color="tab:blue", alpha=0.7)
    axes4[1].set_xticks(x_pos)
    axes4[1].set_xticklabels(labels_short, rotation=45, ha="right", fontsize=7)
    axes4[1].set_ylabel("Varianz (T²)")
    axes4[1].set_title("Varianz pro B-Komponente (gestapelt)")
    axes4[1].legend(fontsize=9)
    axes4[1].grid(axis="y", alpha=0.3)

    fig4.suptitle(
        f">>> Beste (Gesamtvarianz): {labels_short[best_idx_var]} <<<",
        fontsize=13, fontweight="bold", color="darkred",
    )
    plt.tight_layout()
    plt.savefig("plot_informationsgewinn.png", dpi=150, bbox_inches="tight")
    print("  -> plot_informationsgewinn.png gespeichert")

    # --- 4b: Erweiterte Metriken (dimensionslose, vergleichbare Metriken) ---
    fig4b, axes4b = plt.subplots(2, 3, figsize=(20, 10))

    # Normalisierte Entropie (primäre Metrik!)
    entropies_norm = [ig["entropy_normalized"] for ig in info_gains]
    entropies_norm_plot = [e if np.isfinite(e) else np.nan for e in entropies_norm]
    best_idx_ent = int(np.nanargmax(entropies_norm_plot))
    bar_c_ent = ["gold" if i == best_idx_ent else colors[i] for i in range(n_orient)]
    edge_c_ent = ["red" if i == best_idx_ent else "none" for i in range(n_orient)]
    lw_ent = [3 if i == best_idx_ent else 0 for i in range(n_orient)]

    bars_ent = axes4b[0, 0].bar(x_pos, entropies_norm_plot, color=bar_c_ent,
                                 edgecolor=edge_c_ent, linewidth=lw_ent)
    axes4b[0, 0].set_xticks(x_pos)
    axes4b[0, 0].set_xticklabels(labels_short, rotation=45, ha="right", fontsize=6)
    axes4b[0, 0].set_ylabel("Norm. Entropie (dimensionslos)")
    axes4b[0, 0].set_title(f"[EMPFOHLEN] Normalisierte Entropie [Beste: {labels_short[best_idx_ent]}]",
                            fontsize=10, fontweight="bold")
    axes4b[0, 0].grid(axis="y", alpha=0.3)

    # SNR (Variationskoeffizient)
    snrs = [ig["snr"] for ig in info_gains]
    best_idx_snr = int(np.argmax(snrs))
    bar_c_snr = ["gold" if i == best_idx_snr else colors[i] for i in range(n_orient)]
    edge_c_snr = ["red" if i == best_idx_snr else "none" for i in range(n_orient)]
    lw_snr = [3 if i == best_idx_snr else 0 for i in range(n_orient)]

    bars_snr = axes4b[0, 1].bar(x_pos, snrs, color=bar_c_snr,
                                 edgecolor=edge_c_snr, linewidth=lw_snr)
    axes4b[0, 1].set_xticks(x_pos)
    axes4b[0, 1].set_xticklabels(labels_short, rotation=45, ha="right", fontsize=6)
    axes4b[0, 1].set_ylabel("SNR = Std(|B|)/Mean(|B|)")
    axes4b[0, 1].set_title(f"Signal-to-Noise Ratio  [Beste: {labels_short[best_idx_snr]}]",
                            fontsize=10)
    axes4b[0, 1].grid(axis="y", alpha=0.3)

    # Normalisierte Gesamtvarianz
    norm_vars = [ig["normalized_var"] for ig in info_gains]
    best_idx_nv = int(np.argmax(norm_vars))
    bar_c_nv = ["gold" if i == best_idx_nv else colors[i] for i in range(n_orient)]
    edge_c_nv = ["red" if i == best_idx_nv else "none" for i in range(n_orient)]
    lw_nv = [3 if i == best_idx_nv else 0 for i in range(n_orient)]

    bars_nv = axes4b[0, 2].bar(x_pos, norm_vars, color=bar_c_nv,
                                edgecolor=edge_c_nv, linewidth=lw_nv)
    axes4b[0, 2].set_xticks(x_pos)
    axes4b[0, 2].set_xticklabels(labels_short, rotation=45, ha="right", fontsize=6)
    axes4b[0, 2].set_ylabel("Norm. Varianz (dimensionslos)")
    axes4b[0, 2].set_title(f"Normalisierte Varianz  [Beste: {labels_short[best_idx_nv]}]",
                            fontsize=10)
    axes4b[0, 2].grid(axis="y", alpha=0.3)

    # Effektive Dimensionalität (1-3)
    eff_dims = [ig["effective_dim"] for ig in info_gains]
    best_idx_ed = int(np.argmax(eff_dims))
    bar_c_ed = ["gold" if i == best_idx_ed else colors[i] for i in range(n_orient)]
    edge_c_ed = ["red" if i == best_idx_ed else "none" for i in range(n_orient)]
    lw_ed = [3 if i == best_idx_ed else 0 for i in range(n_orient)]

    bars_ed = axes4b[1, 0].bar(x_pos, eff_dims, color=bar_c_ed,
                                edgecolor=edge_c_ed, linewidth=lw_ed)
    axes4b[1, 0].set_xticks(x_pos)
    axes4b[1, 0].set_xticklabels(labels_short, rotation=45, ha="right", fontsize=6)
    axes4b[1, 0].set_ylabel("Effektive Dimensionalität")
    axes4b[1, 0].set_ylim([0, 3.5])
    axes4b[1, 0].axhline(y=3.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Max (isotrop)')
    axes4b[1, 0].set_title(f"Effektive Dim. (3=alle Achsen)  [Beste: {labels_short[best_idx_ed]}]",
                            fontsize=10)
    axes4b[1, 0].grid(axis="y", alpha=0.3)
    axes4b[1, 0].legend(fontsize=7)

    # Volumetrische Effizienz (0-1)
    vol_effs = [ig["volumetric_efficiency"] for ig in info_gains]
    best_idx_ve = int(np.argmax(vol_effs))
    bar_c_ve = ["gold" if i == best_idx_ve else colors[i] for i in range(n_orient)]
    edge_c_ve = ["red" if i == best_idx_ve else "none" for i in range(n_orient)]
    lw_ve = [3 if i == best_idx_ve else 0 for i in range(n_orient)]

    bars_ve = axes4b[1, 1].bar(x_pos, vol_effs, color=bar_c_ve,
                                edgecolor=edge_c_ve, linewidth=lw_ve)
    axes4b[1, 1].set_xticks(x_pos)
    axes4b[1, 1].set_xticklabels(labels_short, rotation=45, ha="right", fontsize=6)
    axes4b[1, 1].set_ylabel("Volumetrische Effizienz")
    axes4b[1, 1].set_ylim([0, 1.1])
    axes4b[1, 1].axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Perfekt isotrop')
    axes4b[1, 1].set_title(f"Vol. Effizienz (1=isotrop)  [Beste: {labels_short[best_idx_ve]}]",
                            fontsize=10)
    axes4b[1, 1].grid(axis="y", alpha=0.3)
    axes4b[1, 1].legend(fontsize=7)

    # Konditionszahl (logarithmisch)
    cond_nums = [ig["condition_num"] for ig in info_gains]
    cond_plot = [c if np.isfinite(c) else np.nan for c in cond_nums]
    finite_conds = [(c, i) for i, c in enumerate(cond_plot) if np.isfinite(c)]
    if finite_conds:
        best_idx_cond = min(finite_conds, key=lambda x: x[0])[1]
    else:
        best_idx_cond = 0
    bar_c_cond = ["gold" if i == best_idx_cond else colors[i] for i in range(n_orient)]
    edge_c_cond = ["red" if i == best_idx_cond else "none" for i in range(n_orient)]
    lw_cond = [3 if i == best_idx_cond else 0 for i in range(n_orient)]

    bars_cond = axes4b[1, 2].bar(x_pos, cond_plot, color=bar_c_cond,
                                  edgecolor=edge_c_cond, linewidth=lw_cond)
    axes4b[1, 2].set_xticks(x_pos)
    axes4b[1, 2].set_xticklabels(labels_short, rotation=45, ha="right", fontsize=6)
    axes4b[1, 2].set_ylabel("Konditionszahl λ_max/λ_min")
    axes4b[1, 2].set_yscale("log")
    axes4b[1, 2].set_title(f"Konditionszahl (niedrig=isotrop)  [Beste: {labels_short[best_idx_cond]}]",
                            fontsize=10)
    axes4b[1, 2].grid(axis="y", alpha=0.3)

    fig4b.suptitle("Erweiterte Informationsgewinn-Metriken (dimensionslos & vergleichbar)", 
                   fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plot_informationsgewinn_erweitert.png", dpi=150, bbox_inches="tight")
    print("  -> plot_informationsgewinn_erweitert.png gespeichert")

    # --- 4c: Ranking-Tabelle als Plot ---
    fig4c, ax4c = plt.subplots(figsize=(18, max(6, n_orient * 0.35)))
    ax4c.axis("off")

    # Ranking nach NORMALISIERTER Entropie (einheitenunabhängig!)
    entropies_norm = [ig["entropy_normalized"] for ig in info_gains]
    entropies_norm_finite = [e if np.isfinite(e) else -1e30 for e in entropies_norm]
    ranking_idx = np.argsort(entropies_norm_finite)[::-1]  # absteigend
    
    eff_dims = [ig["effective_dim"] for ig in info_gains]
    vol_effs = [ig["volumetric_efficiency"] for ig in info_gains]
    
    table_data = []
    for rank, idx in enumerate(ranking_idx):
        ent_norm_str = f"{entropies_norm[idx]:.4f}" if np.isfinite(entropies_norm[idx]) else "−∞"
        table_data.append([
            f"{rank+1}",
            labels_short[idx],
            ent_norm_str,
            f"{snrs[idx]:.4f}",
            f"{norm_vars[idx]:.4f}",
            f"{eff_dims[idx]:.3f}",
            f"{vol_effs[idx]:.4f}",
            f"{cond_nums[idx]:.1f}" if np.isfinite(cond_nums[idx]) else "∞",
        ])

    col_labels = ["Rang", "Orientierung", "Entropie\n(norm.)", "SNR",
                  "Norm.\nVarianz", "Eff.\nDim.", "Vol.\nEff.", "Kond.\nZahl"]
    table = ax4c.table(cellText=table_data, colLabels=col_labels,
                        loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.3)

    # Header farbig
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Top-3 hervorheben
    for rank in range(min(3, len(ranking_idx))):
        for j in range(len(col_labels)):
            table[rank + 1, j].set_facecolor("#D4E6F1" if rank > 0 else "#A9DFBF")

    ax4c.set_title("[EMPFOHLEN] Ranking nach Normalisierter Entropie (dimensionslos, skalierungsinvariant)",
                    fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("plot_ranking.png", dpi=150, bbox_inches="tight")
    print("  -> plot_ranking.png gespeichert")

    # ----------------------------------------------------------
    # Plot 5: Abstände Sensor ↔ Magnet
    # ----------------------------------------------------------
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    for i, res in enumerate(results):
        ax5.plot(path_idx, res["distances"] * 100, color=colors[i],
                 linewidth=1, label=res["label"])
    ax5.set_xlabel("Pfadindex")
    ax5.set_ylabel("Abstand (cm)")
    ax5.set_title("Abstand Sensor <-> Magnet entlang des Pfades")
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_abstaende.png", dpi=150, bbox_inches="tight")
    print("  -> plot_abstaende.png gespeichert")

    plt.close("all")


# ==============================================================
# ANIMATIONEN
# ==============================================================

def create_animations(results: list[dict]):
    """
    Erstellt interaktive Plotly-Animationen für jede Orientierung
    und speichert sie als HTML-Dateien.
    """
    print("\nErstelle Animationen (Plotly Backend) ...")
    
    for i, res in enumerate(results):
        label = res["label"]
        magnet = res["magnet"]
        
        # Sensor erstellen (fix am Ursprung)
        sensor = magpy.Sensor(position=SENSOR_POS)
        
        # Beschriftung für Dateinamen bereinigen (Windows-kompatibel)
        safe_label = (
            label.replace("/", "-")
            .replace("||", "_")
            .replace("°", "deg")
            .replace("|", "_")
            .replace(":", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "_")
            .replace(">", "")
            .replace("<", "")
            .replace("-", "")
        )
        filename = f"animation_{i+1:02d}_{safe_label[:40]}.html"
        
        print(f"  [{i+1}/{len(results)}] {label} ... ", end="", flush=True)
        
        try:
            # Animation mit Plotly erstellen
            # animation=3 -> 3 Sekunden Animation
            # backend='plotly' -> Interaktiver Plotly-Output
            # return_fig=True -> Figur zurückgeben für Speicherung
            fig = magpy.show(
                magnet,
                sensor,
                animation=3,
                animation_slider=True,
                animation_fps=20,
                backend='plotly',
                return_fig=True,
                style_path_show=True,
                style_legend_show=True,
            )
            
            # Layout anpassen
            fig.update_layout(
                title=dict(
                    text=f"Magnetfeld-Simulation: {label}",
                    x=0.5,
                    xanchor='center',
                    font=dict(size=16, color='darkblue'),
                ),
                scene=dict(
                    xaxis_title='X (m)',
                    yaxis_title='Y (m)',
                    zaxis_title='Z (m)',
                    aspectmode='data',
                ),
                width=1200,
                height=800,
            )
            
            # Als HTML speichern
            fig.write_html(
                filename,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'magnet_animation_{i+1}',
                        'height': 800,
                        'width': 1200,
                        'scale': 2,
                    },
                },
            )
            
            print(f"-> {filename}")
            
        except Exception as e:
            print(f"FEHLER: {e}")
            continue
    
    print("Alle Animationen erstellt.")


# ==============================================================
# HAUPTPROGRAMM
# ==============================================================

def main():
    print("=" * 60)
    print("  Magpylib Magnetfeld-Simulation")
    print("=" * 60)
    print(f"  Magnet:     Zylinder D={DIAMETER*1e3:.0f} mm, H={HEIGHT*1e3:.0f} mm")
    print(f"  Magnetis.:  M = {MAG_VECTOR} A/m  (J ~ {4*np.pi*1e-7*MAG_VECTOR[2]:.3f} T)")
    print(f"  Sensor:     {SENSOR_POS} m")
    print(f"  Magnet0:    {MAGNET_INIT_POS} m")
    print(f"  Drehpunkt:  {PIVOT} m")
    print(f"  Schwingung: +/-{SWING_ANGLE} deg, {N_PATH} Pfadpunkte")
    print(f"  Varianten:  {len(ORIENTATIONS)} Orientierungen")
    print()

    # Pendelwinkel generieren
    angles = generate_pendulum_angles(N_HALF, SWING_ANGLE)
    print(f"  Pfadwinkel: {len(angles)} Punkte  [{angles.min():.1f} .. {angles.max():.1f}] deg")

    # ==========================================================
    # Simulation für alle Orientierungen
    # ==========================================================
    results: list[dict] = []
    info_gains: list[dict] = []

    for orient_label, orient_rot in ORIENTATIONS:
        print(f"\n  Simuliere: {orient_label} ... ", end="", flush=True)
        res = simulate_single_orientation(orient_label, orient_rot, angles)
        ig = compute_info_gain(res["B"])
        results.append(res)
        info_gains.append(ig)

        B_norms = np.linalg.norm(res["B"], axis=1)
        print(f"|B| = [{B_norms.min():.3e}, {B_norms.max():.3e}] T,  "
              f"Var_total = {ig['total_var']:.3e}")

    # ==========================================================
    # Plausibilitätschecks
    # ==========================================================
    plausibility_checks(results)

    # ==========================================================
    # Beste Ausrichtung ermitteln (multiple Metriken)
    # ==========================================================
    total_vars = [ig["total_var"] for ig in info_gains]
    entropies = [ig["entropy"] for ig in info_gains]
    entropies_norm = [ig["entropy_normalized"] for ig in info_gains]
    snrs = [ig["snr"] for ig in info_gains]
    eff_dims = [ig["effective_dim"] for ig in info_gains]
    vol_effs = [ig["volumetric_efficiency"] for ig in info_gains]

    best_idx_var = int(np.argmax(total_vars))
    
    # Normalisierte Entropie als primäre Metrik (skalierungsinvariant!)
    entropies_norm_finite = [e if np.isfinite(e) else -1e30 for e in entropies_norm]
    best_idx_ent = int(np.argmax(entropies_norm_finite))
    
    best_idx_snr = int(np.argmax(snrs))
    best_idx_eff_dim = int(np.argmax(eff_dims))
    best_idx_vol_eff = int(np.argmax(vol_effs))

    print(f"\n{'='*70}")
    print(f"  ERGEBNISSE - INFORMATIONSGEWINN (DIMENSIONSLOSE METRIKEN)")
    print(f"{'='*70}")
    
    print(f"\n  [EMPFOHLEN] Normalisierte Entropie: {ORIENTATIONS[best_idx_ent][0]}")
    print(f"    Entropie (norm.):  {entropies_norm[best_idx_ent]:>8.4f}")
    print(f"    Eff. Dimension:    {eff_dims[best_idx_ent]:>8.4f} / 3.0  (Achsen mit signifikanter Variation)")
    print(f"    Vol. Effizienz:    {vol_effs[best_idx_ent]:>8.4f}       (1.0 = perfekt isotrop)")
    print(f"    SNR:               {snrs[best_idx_ent]:>8.4f}       (Variationskoeffizient)")

    print(f"\n  Alternative Kriterien:")
    print(f"  {'-'*66}")
    print(f"    Max. Gesamtvarianz:    {ORIENTATIONS[best_idx_var][0]}")
    print(f"      -> Total_var = {total_vars[best_idx_var]:.3e} T\u00b2")
    print(f"\n    Max. SNR:              {ORIENTATIONS[best_idx_snr][0]}")
    print(f"      -> SNR = {snrs[best_idx_snr]:.6f}")
    print(f"\n    Max. Eff. Dimension:   {ORIENTATIONS[best_idx_eff_dim][0]}")
    print(f"      -> {eff_dims[best_idx_eff_dim]:.4f} / 3.0 Achsen")
    print(f"\n    Max. Vol. Effizienz:   {ORIENTATIONS[best_idx_vol_eff][0]}")
    print(f"      -> {vol_effs[best_idx_vol_eff]:.4f} (Isotropie)")

    # Top-5 Ranking nach normalisierter Entropie
    ranking = np.argsort(entropies_norm_finite)[::-1]
    print(f"\n  {'-'*66}")
    print(f"  Top-5 Ranking (Normalisierte Entropie - skalierungsinvariant):")
    print(f"  {'-'*66}")
    print(f"  {'Rang':<5} {'Orientierung':<30} {'H_norm':>8} {'Eff.D.':>7} {'SNR':>8}")
    print(f"  {'-'*66}")
    for rank, idx in enumerate(ranking[:5]):
        ent_str = f"{entropies_norm[idx]:.4f}" if np.isfinite(entropies_norm[idx]) else "-inf"
        print(f"  {rank+1:<5} {ORIENTATIONS[idx][0]:<30} {ent_str:>8} {eff_dims[idx]:>7.3f} {snrs[idx]:>8.4f}")
    print(f"  {'-'*66}")
    print(f"{'='*70}")

    # ==========================================================
    # Plots erstellen und speichern
    # ==========================================================
    print("\nErstelle Plots ...")
    plot_results(results, info_gains)
    print("\nAlle Plots gespeichert.")

    # ==========================================================
    # Animationen erstellen
    # ==========================================================
    create_animations(results)
    
    print("\nSimulation abgeschlossen.")


if __name__ == "__main__":
    main()
