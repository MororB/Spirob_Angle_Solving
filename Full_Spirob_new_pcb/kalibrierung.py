import numpy as np
import cv2
import glob
import os

# --- KONFIGURATION ---
# Anzahl der INNEREN Ecken des Schachbretts (nicht die Quadrate!)
# Bei einem Standard 10x7 Brett sind es meistens 9x6 innere Ecken.
CHECKERBOARD = (9, 6) 
SQUARE_SIZE = 25  # Größe eines Quadrats in mm (für echte Maßstäbe wichtig, hier optional)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Listen für 3D-Punkte (Welt) und 2D-Punkte (Bild)
objpoints = [] 
imgpoints = [] 

# Koordinaten des Schachbretts im 3D-Raum definieren (z.B. (0,0,0), (1,0,0), (2,0,0) ...)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Deine Kamera

print("Starte Kalibrierung...")
print("Drücke 'c', um ein Bild aufzunehmen (sammle min. 10 Stück).")
print("Drücke 'q', um die Kalibrierung zu berechnen und zu beenden.")

count = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    copy_frame = frame.copy()

    # Schachbrett finden
    ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # Wenn gefunden, zeichnen
    if ret_corners:
        cv2.drawChessboardCorners(copy_frame, CHECKERBOARD, corners, ret_corners)
        cv2.putText(copy_frame, "Brett erkannt! Druecke 'c'", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(copy_frame, f"Bilder: {count}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Calibration', copy_frame)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('c') and ret_corners:
        objpoints.append(objp)
        
        # Ecken präzisieren (Subpixel-Genauigkeit)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        count += 1
        print(f"Bild {count} gespeichert.")
        
    elif key & 0xFF == ord('q'):
        if count < 5:
            print("Zu wenig Bilder! Bitte mindestens 5, besser 10-20 aufnehmen.")
        else:
            break

cap.release()
cv2.destroyAllWindows()

if count >= 5:
    print("Berechne Kalibrierung (das kann kurz dauern)...")
    # Die eigentliche Mathematik passiert hier:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Kameramatrix:\n", mtx)
    print("Verzerrungskoeffizienten:\n", dist)

    # Speichern in eine Datei für dein Live-Skript
    np.savez("calibration_data.npz", mtx=mtx, dist=dist)
    print("Kalibrierung gespeichert als 'calibration_data.npz'!")