import cv2
import numpy as np
import math
import os

def wrap_deg(a):
    return (a + 180) % 360 - 180   # -> [-180, 180)


# ==================== ROBUSTE 3D POSE ESTIMATION ====================

def get_marker_obj_points(marker_size):
    """3D-Eckpunkte eines Markers im Marker-Koordinatensystem (Z=0 Ebene)."""
    half = marker_size / 2.0
    return np.array([
        [-half,  half, 0],   # Corner 0: top-left
        [ half,  half, 0],   # Corner 1: top-right
        [ half, -half, 0],   # Corner 2: bottom-right
        [-half, -half, 0],   # Corner 3: bottom-left
    ], dtype=np.float64)


def estimate_marker_rotation(corner_2d, marker_size, cam_matrix, dist_coeffs, prev_R=None):
    """
    Robuste 3D-Pose-Schätzung mit Ambiguitäts-Behandlung und temporaler Stabilisierung.
    
    Problem: Ein flacher quadratischer Marker hat ZWEI gültige 3D-Posen
    (er könnte "von vorn" oder "von hinten" gesehen werden).
    
    Lösung: solvePnPGeneric + IPPE_SQUARE gibt beide Lösungen.
    Wir wählen die, die am besten passt basierend auf:
    1. Geometrische Prüfung (Normale zur Kamera)
    2. Temporale Konsistenz (ähnlich zur vorherigen Pose)
    3. Reprojektionsfehler
    """
    obj_pts = get_marker_obj_points(marker_size)
    img_pts = corner_2d.reshape(-1, 1, 2).astype(np.float64)
    
    try:
        n_solutions, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(
            obj_pts, img_pts, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
    except Exception:
        return None, None, None
    
    if n_solutions == 0:
        return None, None, None
    
    # Bewerte alle Lösungen
    candidates = []
    
    for i in range(n_solutions):
        R, _ = cv2.Rodrigues(rvecs[i])
        err = reproj_errors[i][0]
        
        # Score-Komponenten
        score = 0
        
        # 1. Geometrie-Check: Marker vor Kamera, Normale zur Kamera
        # In OpenCV: Kamera-Z zeigt in Szene (+Z), Marker-Z sollte zur Kamera zeigen (-Z)
        # Also: R[2,2] < 0 (Marker-Z hat negative Z-Komponente in Kamera-Koordinaten)
        marker_in_front = tvecs[i][2, 0] > 0.01  # mindestens 1cm entfernt
        normal_toward_camera = R[2, 2] < -0.1  # deutlich zur Kamera gerichtet
        
        if marker_in_front and normal_toward_camera:
            score += 1000  # Sehr stark bevorzugen - das ist die geometrisch korrekte Lösung
        
        # 2. Temporale Konsistenz: Wie ähnlich zur vorherigen Pose?
        if prev_R is not None:
            # Frobenius-Norm der Rotationsdifferenz
            diff = np.linalg.norm(R - prev_R, 'fro')
            # Je kleiner der Unterschied, desto besser (invertiert)
            # Typisch: diff < 0.5 = sehr ähnlich, diff > 2 = sehr verschieden
            consistency_score = max(0, 50 - diff * 20)
            score += consistency_score
        
        # 3. Reprojektionsfehler (invertiert: kleiner = besser)
        # Typisch: err < 1.0 = gut, err > 5.0 = schlecht
        score -= err * 10
        
        candidates.append({
            'R': R,
            'rvec': rvecs[i],
            'tvec': tvecs[i],
            'score': score,
            'error': err,
            'geom_ok': marker_in_front and normal_toward_camera
        })
    
    # Wähle Lösung mit höchstem Score
    best = max(candidates, key=lambda x: x['score'])
    
    # Debug: Wenn geometrisch falsche Lösung gewählt wurde, warnen
    if not best['geom_ok'] and any(c['geom_ok'] for c in candidates):
        # Es gibt eine geometrisch korrekte Lösung, aber wir haben eine falsche gewählt
        # Das sollte nicht passieren mit den hohen Scores (1000 Punkte)
        pass
    
    return best['R'], best['rvec'], best['tvec']


def compute_joint_angle_3d(R_a, R_b):
    """
    Berechnet den Gelenkwinkel zwischen zwei Markern aus ihren 3D-Rotationsmatrizen.
    
    Robuste Methode: Projiziert beide Marker-X-Achsen auf eine gemeinsame Ebene
    (die Ebene senkrecht zur mittleren Normalen) und berechnet den Winkel zwischen den Projektionen.
    
    Das ist unabhängig vom Kamerawinkel, da wir die 3D-Orientierungen verwenden.
    """
    # Extrahiere X-Achsen (erste Spalte von R) - zeigen entlang der Marker-Kante
    x_a = R_a[:, 0]
    x_b = R_b[:, 0]
    
    # Extrahiere Z-Achsen (Normalen) - zeigen von Marker-Oberfläche weg
    z_a = R_a[:, 2]
    z_b = R_b[:, 2]
    
    # Mittlere Normale (gemeinsame Referenzebene)
    z_mean = (z_a + z_b) / 2
    z_mean = z_mean / np.linalg.norm(z_mean)
    
    # Projiziere X-Achsen auf Ebene senkrecht zu z_mean
    # Projektion: v_proj = v - (v·n)n
    x_a_proj = x_a - np.dot(x_a, z_mean) * z_mean
    x_b_proj = x_b - np.dot(x_b, z_mean) * z_mean
    
    # Normalisieren
    x_a_proj = x_a_proj / np.linalg.norm(x_a_proj)
    x_b_proj = x_b_proj / np.linalg.norm(x_b_proj)
    
    # Winkel zwischen projizierten Vektoren
    # atan2(cross, dot) gibt vorzeichenbehafteten Winkel
    cross = np.cross(x_a_proj, x_b_proj)
    dot = np.dot(x_a_proj, x_b_proj)
    
    # Vorzeichen von cross·z_mean gibt Rotationsrichtung
    angle = math.atan2(np.dot(cross, z_mean), dot)
    
    return wrap_deg(math.degrees(angle))


# ==================== KALIBRIERUNG LADEN ====================

CALIB_FILE = "calibration_data.npz"
calibration_loaded = False

if os.path.exists(CALIB_FILE):
    data = np.load(CALIB_FILE)
    mtx = data['mtx']
    dist = data['dist']
    calibration_loaded = True
    print(f"✅ Kalibrierung geladen aus {CALIB_FILE}")
    print(f"   3D Pose Estimation aktiviert (perspektiv-korrigiert)")
else:
    mtx = None
    dist = None
    print("⚠️  Keine 'calibration_data.npz' gefunden!")
    print("   Fallback: 2D-Winkelmessung (nur korrekt bei frontaler Kamera)")


# ==================== KAMERA & DETEKTOR ====================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

win = "ArUco Test"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 1920, 1080)

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
params = aruco.DetectorParameters()

# Optimierte Parameter
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
params.cornerRefinementMaxIterations = 30
params.cornerRefinementMinAccuracy = 0.01
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 23
params.adaptiveThreshWinSizeStep = 10
params.minMarkerPerimeterRate = 0.03
params.maxMarkerPerimeterRate = 4.0
params.polygonalApproxAccuracyRate = 0.05
params.errorCorrectionRate = 0.6
params.perspectiveRemovePixelPerCell = 4
params.perspectiveRemoveIgnoredMarginPerCell = 0.13
params.minCornerDistanceRate = 0.05
params.minDistanceToBorder = 3

detector = aruco.ArucoDetector(dictionary, params)


# ==================== KONFIGURATION ====================

order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # 14 Marker -> 13 Gelenke

# Marker-Größe in Metern (Seitenlänge des schwarzen Quadrats messen!)
MARKER_SIZE = 0.012  # 1.2cm - WICHTIG: Physisch nachmessen!

# Temporal smoothing für stabile Posen
SMOOTHING_ALPHA = 0.3  # 0 = keine Glättung, 1 = nur alte Werte
prev_rotations = {}  # id -> R (letzte gültige Rotation)
prev_joint_angles = {}  # joint_index -> angle

# Undistortion
newcameramtx = None
roi = None

# Modus: 3D oder 2D (Taste 'm' zum Umschalten)
use_3d_mode = calibration_loaded
undistort_enabled = calibration_loaded

print(f"\n🎮 Steuerung:")
print(f"   ESC  = Beenden")
print(f"   m    = Umschalten 2D/3D Modus")
print(f"   u    = Undistortion an/aus")
print(f"   +/-  = Marker-Größe anpassen ({MARKER_SIZE*1000:.1f}mm)")
print(f"   s    = Smoothing anpassen (aktuell: {SMOOTHING_ALPHA:.2f})")
print(f"\n⚠️  WICHTIG: Marker-Größe physisch nachmessen!")
print(f"   Aktuell: {MARKER_SIZE*1000:.1f}mm (schwarzes Quadrat)")
print(f"   Bei falscher Größe stimmen 3D-Winkel nicht!\n")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # --- ENTZERRUNG ---
    if calibration_loaded and undistort_enabled:
        h, w = frame.shape[:2]
        if newcameramtx is None:
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        processing_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    else:
        processing_frame = frame

    # Kameramatrix für Pose (nach Undistortion: newcameramtx, dist=0)
    cam_mtx = newcameramtx if (calibration_loaded and undistort_enabled and newcameramtx is not None) else mtx
    dist_for_pose = np.zeros(5) if (calibration_loaded and undistort_enabled) else dist

    gray = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    rotations = {}   # id -> R (3x3) oder float (2D Winkel)
    centers = {}     # id -> (cx, cy)

    if ids is not None and len(ids) > 0:
        ids_flat = ids.flatten()
        aruco.drawDetectedMarkers(processing_frame, corners, ids)

        for corner, mid in zip(corners, ids_flat):
            mid = int(mid)
            cx = int(np.mean(corner[0][:, 0]))
            cy = int(np.mean(corner[0][:, 1]))
            centers[mid] = (cx, cy)

            if use_3d_mode and cam_mtx is not None:
                # === 3D: Robuste Pose-Estimation mit temporaler Stabilisierung ===
                prev_R = prev_rotations.get(mid, None)
                R, rvec, tvec = estimate_marker_rotation(
                    corner[0], MARKER_SIZE, cam_mtx, dist_for_pose, prev_R
                )
                if R is not None:
                    # Temporal smoothing der Rotation
                    if prev_R is not None:
                        R = (1 - SMOOTHING_ALPHA) * R + SMOOTHING_ALPHA * prev_R
                        # Re-orthogonalisierung nach Mittelung
                        U, _, Vt = np.linalg.svd(R)
                        R = U @ Vt
                    
                    rotations[mid] = R
                    prev_rotations[mid] = R
                    
                    # Rvec für Visualisierung neu berechnen
                    rvec, _ = cv2.Rodrigues(R)
                    
                    cv2.drawFrameAxes(processing_frame, cam_mtx, dist_for_pose,
                                      rvec, tvec, MARKER_SIZE * 0.5)
                    z_rot = math.degrees(math.atan2(R[1, 0], R[0, 0]))
                    cv2.putText(processing_frame, f"ID{mid} z:{z_rot:.0f}",
                                (cx - 30, cy - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            else:
                # === 2D: Bildebenen-Winkel ===
                p0 = corner[0][0]
                p1 = corner[0][1]
                alpha = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
                rotations[mid] = alpha
                cv2.putText(processing_frame, f"ID{mid} {alpha:.0f}",
                            (cx - 30, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # === GELENKWINKEL ===
        for i in range(len(order) - 1):
            a = order[i]
            b = order[i + 1]

            if a in rotations and b in rotations:
                if use_3d_mode and isinstance(rotations[a], np.ndarray):
                    theta_deg = compute_joint_angle_3d(rotations[a], rotations[b])
                else:
                    theta_deg = wrap_deg(rotations[b] - rotations[a])
                
                # Temporal smoothing der Gelenkwinkel
                joint_key = f"{a}_{b}"
                if joint_key in prev_joint_angles:
                    theta_deg = (1 - SMOOTHING_ALPHA) * theta_deg + SMOOTHING_ALPHA * prev_joint_angles[joint_key]
                prev_joint_angles[joint_key] = theta_deg

                x = int((centers[a][0] + centers[b][0]) / 2)
                y = int((centers[a][1] + centers[b][1]) / 2)

                cv2.putText(processing_frame, f"J{i+1}: {theta_deg:.1f}", (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(processing_frame, f"J{i+1}: {theta_deg:.1f}", (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Status
    mode_str = "3D" if use_3d_mode else "2D"
    undist_str = "ON" if undistort_enabled else "OFF"
    cv2.putText(processing_frame, f"Modus: {mode_str} | Undist: {undist_str} | Smooth: {SMOOTHING_ALPHA:.2f} | Size: {MARKER_SIZE*1000:.1f}mm",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(processing_frame, f"m=Modus u=Undist +/-=Size s=Smooth ESC=Ende",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    disp_x, disp_y, disp_w, disp_h = cv2.getWindowImageRect(win)
    if disp_w > 0 and disp_h > 0:
        frame_show = cv2.resize(processing_frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
    else:
        frame_show = processing_frame

    cv2.imshow(win, frame_show)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('m'):
        if calibration_loaded:
            use_3d_mode = not use_3d_mode
            prev_rotations.clear()  # Reset smoothing
            prev_joint_angles.clear()
            print(f"Modus: {'3D (perspektiv-korrigiert)' if use_3d_mode else '2D (Bildebene)'}")
        else:
            print("3D-Modus benötigt Kalibrierungsdaten!")
    elif key == ord('u'):
        if calibration_loaded:
            undistort_enabled = not undistort_enabled
            newcameramtx = None
            print(f"Undistortion: {'AN' if undistort_enabled else 'AUS'}")
    elif key == ord('+') or key == ord('='):
        MARKER_SIZE += 0.0005  # +0.5mm
        prev_rotations.clear()
        print(f"Marker-Größe: {MARKER_SIZE*1000:.1f}mm")
    elif key == ord('-') or key == ord('_'):
        MARKER_SIZE = max(0.005, MARKER_SIZE - 0.0005)  # -0.5mm, min 5mm
        prev_rotations.clear()
        print(f"Marker-Größe: {MARKER_SIZE*1000:.1f}mm")
    elif key == ord('s'):
        SMOOTHING_ALPHA = (SMOOTHING_ALPHA + 0.1) % 1.0
        print(f"Smoothing: {SMOOTHING_ALPHA:.2f} (0=keine Glättung, 0.9=stark)")

cap.release()
cv2.destroyAllWindows()