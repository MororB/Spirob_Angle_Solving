import cv2
import numpy as np
import math
import os

def wrap_deg(a):
    """ Normiert den Winkel auf den Bereich [-180, 180) """
    return (a + 180) % 360 - 180

# --- KALIBRIERUNG LADEN ---
CALIB_FILE = "calibration_data.npz"
calibration_loaded = False

if os.path.exists(CALIB_FILE):
    data = np.load(CALIB_FILE)
    mtx = data['mtx']   # Kameramatrix
    dist = data['dist'] # Verzerrungskoeffizienten
    calibration_loaded = True
    print(f"Kalibrierung geladen aus {CALIB_FILE}")
else:
    print("WARNUNG: Keine 'calibration_data.npz' gefunden!")
    print("Das Bild wird NICHT entzerrt. Bitte erst das Kalibrierungs-Skript ausführen.")

# --- INITIALISIERUNG ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

win = "ArUco Multi-Segment Tracking (14 Segments)"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 1280, 720)

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, params)

# NEU: 14 Segmente ergeben IDs von 0 bis 13
order = list(range(14)) 

newcameramtx = None
roi = None

START_WITH_UNDISTORT = True # Set to False to start without undistortion
undistort_enabled = calibration_loaded and START_WITH_UNDISTORT

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # --- ENTZERRUNG ---
    if calibration_loaded and undistort_enabled:
        h, w = frame.shape[:2]
        if newcameramtx is None:
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        processing_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    else:
        processing_frame = frame.copy()

    # Status-Anzeige für Entzerrung
    status_text = f"Undistort: {'ON' if undistort_enabled else 'OFF'} ('u' to toggle)"
    if not calibration_loaded:
        status_text += " (No Calib Data)"
    cv2.putText(processing_frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if undistort_enabled else (0, 0, 255), 2)

    gray = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    angles = {}   # id -> alpha (deg)
    centers = {}  # id -> (cx, cy)

    if ids is not None and len(ids) > 0:
        ids_flat = ids.flatten()
        aruco.drawDetectedMarkers(processing_frame, corners, ids)

        # 1. Marker-Daten extrahieren
        for c, mid in zip(corners, ids_flat):
            # Winkel basierend auf der Oberkante des Markers (Ecke 0 zu Ecke 1)
            p0 = c[0][0]
            p1 = c[0][1]
            alpha = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))

            # Zentrum des Markers
            cx = int(np.mean(c[0][:, 0]))
            cy = int(np.mean(c[0][:, 1]))

            angles[int(mid)] = alpha
            centers[int(mid)] = (cx, cy)

            # Kompakte ID-Anzeige direkt am Marker
            cv2.putText(processing_frame, f"ID {int(mid)}", (cx - 50, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # 2. Gelenkwinkel berechnen (14 Segmente -> 13 Gelenke)
        for i in range(len(order) - 1):
            id_a = order[i]
            id_b = order[i + 1]

            if id_a in angles and id_b in angles:
                # Relativer Winkel (Joint Angle)
                theta = wrap_deg(angles[id_b] - angles[id_a])

                # Position für die Anzeige (Mittig zwischen zwei Markern)
                x_text = int((centers[id_a][0] + centers[id_b][0]) / 2)
                y_text = int((centers[id_a][1] + centers[id_b][1]) / 2)

                # Anzeige Joint J1, J2, ... J13
                label = f"J{i+1}: {theta:.1f}"
                cv2.putText(processing_frame, label, (x_text, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3) # Outline
                cv2.putText(processing_frame, label, (x_text, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1) # Text
    
    # --- ANZEIGE ---
    # Fenstergröße anpassen (behält das Seitenverhältnis bei der Skalierung)
    disp_x, disp_y, disp_w, disp_h = cv2.getWindowImageRect(win)
    if disp_w > 0 and disp_h > 0:
        frame_show = cv2.resize(processing_frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
    else:
        frame_show = processing_frame

    cv2.imshow(win, frame_show)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC zum Beenden
        break
    elif key == ord('u'): # 'u' zum Umschalten der Entzerrung
        if calibration_loaded:
            undistort_enabled = not undistort_enabled
            print(f"Entzerrung: {'AN' if undistort_enabled else 'AUS'}")
        else:
            print("Keine Kalibrierungsdaten vorhanden. Entzerrung kann nicht aktiviert werden.")

cap.release()
cv2.destroyAllWindows()