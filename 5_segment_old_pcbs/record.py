import cv2, numpy as np, math, csv, time, threading, struct
import serial
import os
from pathlib import Path
from typing import NamedTuple

# ===================== Konfig =====================
RUN_NAME = time.strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = Path("runs") / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

# DATEINAME: Exakt die Datei aus deinem ersten Schritt!
CALIB_FILE = "calibration_data.npz"

# Serielle Konfiguration
SERIAL_PORT = "COM5" 
BAUD = 1_000_000
SERIAL_TIMEOUT = 0.2

# Kamera Konfiguration
CAM_IDX = 0
BACKEND = cv2.CAP_DSHOW # DSHOW ist unter Windows oft schneller/stabiler für Settings
WIN = "Recording (Undistorted)"

# ACHTUNG: Das muss zur Kalibrierung passen!
# Wenn du beim Kalibrieren nichts eingestellt hast, war es 640x480.
REQUESTED_WIDTH = 640   
REQUESTED_HEIGHT = 480
REQUESTED_FPS = 30

# Marker & Sensor Konfiguration
NUM_CHIPS = 5
N_SEGMENTS = 5
N_ANGLES = N_SEGMENTS - 1 

ID_TO_SEGMENT = {0:0, 1:1, 2:2, 3:3, 4:4}

# ===================== ESP32 Serial Protokoll =====================
HEADER = b"\xAA\x55"
TXYZ_SIZE = 16 
BOARD_PAYLOAD = NUM_CHIPS * TXYZ_SIZE

class SensorData(NamedTuple):
    id: int
    t: float
    x: float
    y: float
    z: float

def read_exact_robust(ser, n, timeout_sec=0.1):
    buf = bytearray()
    t_start = time.time()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if chunk:
            buf.extend(chunk)
        if (time.time() - t_start) > timeout_sec:
            return None
    return bytes(buf)

def next_board_frame(ser):
    while True:
        b = ser.read(1)
        if not b: return None 
        if b == HEADER[:1]:
            b2 = ser.read(1)
            if b2 == HEADER[1:]: break
    
    bid_b = ser.read(1)
    if not bid_b: return None
    board_id = bid_b[0]

    payload = read_exact_robust(ser, BOARD_PAYLOAD, timeout_sec=0.05)
    if payload is None: return None

    sensors = []
    for i in range(NUM_CHIPS):
        chunk = payload[i*TXYZ_SIZE:(i+1)*TXYZ_SIZE]
        t_s, x, y, z = struct.unpack("<ffff", chunk)
        sensors.append(SensorData(i, t_s, x, y, z))
    return board_id, sensors

# ===================== ArUco Winkel =====================
def wrap_pi(a):
    return (a + math.pi) % (2*math.pi) - math.pi

def marker_angle_from_corners(corners_4x2):
    p0 = corners_4x2[0]
    p1 = corners_4x2[1]
    return math.atan2(p1[1] - p0[1], p1[0] - p0[0])

# ===================== Serial Logger =====================
def serial_logger(stop_flag, start_event, out_csv):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=SERIAL_TIMEOUT)
        ser.reset_input_buffer()

        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_ns","board_id","sensor_id","t","x","y","z"])
            print(f"[serial] Logger bereit auf {SERIAL_PORT}")

            while not stop_flag.is_set():
                frm = next_board_frame(ser)
                if frm is None: continue

                board_id, sensors = frm
                t_ns = time.perf_counter_ns()

                for sensor in sensors:
                    if start_event.is_set():
                        w.writerow([t_ns, board_id, sensor.id, sensor.t, sensor.x, sensor.y, sensor.z])
                
        ser.close()
    except Exception as e:
        print(f"[serial] ERROR: {e}")
    finally:
        stop_flag.set()

# ===================== Main =====================
def main():
    # --- 1. Kalibrierung laden ---
    mtx, dist = None, None
    newcameramtx, roi = None, None
    
    if os.path.exists(CALIB_FILE):
        # Laden der .npz Datei (Keywords: mtx, dist)
        data = np.load(CALIB_FILE)
        
        # Manchmal heißen die Keys anders, wir probieren die üblichen:
        if 'mtx' in data: mtx = data['mtx']
        elif 'camera_matrix' in data: mtx = data['camera_matrix']
        
        if 'dist' in data: dist = data['dist']
        elif 'dist_coeffs' in data: dist = data['dist_coeffs']
        
        if mtx is not None and dist is not None:
            print(f"[Main] Kalibrierung geladen: {CALIB_FILE}")
            print(f"       Matrix Shape: {mtx.shape}")
        else:
            print("[Main] FEHLER: Konnte mtx/dist nicht aus npz lesen!")
    else:
        print(f"[Main] WARNUNG: Datei '{CALIB_FILE}' nicht gefunden! Aufnahme läuft verzerrt.")

    # --- 2. Threads starten ---
    stop_flag = threading.Event()
    start_recording_event = threading.Event()
    th = threading.Thread(target=serial_logger, args=(stop_flag, start_recording_event, RUN_DIR/"sensors.csv"))
    th.start()
    
    # --- 3. Kamera Setup ---
    cap = cv2.VideoCapture(CAM_IDX, BACKEND)
    # WICHTIG: Setze die Auflösung BEVOR wir lesen
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, REQUESTED_FPS)

    # Prüfen was wir bekommen haben
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    print(f"[Camera] {W}x{H} @ {FPS} FPS")

    # --- 4. UI Setup ---
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    
    # ArUco
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    use_new = hasattr(aruco, "ArucoDetector")
    detector = aruco.ArucoDetector(dictionary, params) if use_new else None

    # VideoWriter (Platzhalter)
    out = None 

    # --- 5. Preview Loop ---
    print("Drücke ENTER im Fenster zum Starten!")
    while True:
        ok, frame = cap.read()
        if not ok: break

        # --- ENTZERRUNG ---
        if mtx is not None:
            # Wir berechnen die neue Matrix nur 1x basierend auf der tatsächlichen Framegröße
            if newcameramtx is None:
                h, w = frame.shape[:2]
                # alpha=1: Alle Pixel behalten (Schwarze Ränder)
                # alpha=0: Zuschneiden (Keine Ränder)
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            
            # Undistort
            frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        # ------------------
        
        cv2.putText(frame, "ENTER to record | ESC to quit", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        if mtx is not None:
             cv2.putText(frame, f"UNDISTORTED ({CALIB_FILE})", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 13: # Enter
            start_recording_event.set()
            break
        elif key == 27: # Esc
            stop_flag.set()
            th.join()
            return

    # --- 6. Recording Loop ---
    # CSV Header
    with open(RUN_DIR/"frames.csv","w",newline="") as ff, \
         open(RUN_DIR/"labels.csv","w",newline="") as lf:
        
        fw = csv.writer(ff)
        lw = csv.writer(lf)
        
        fw.writerow(["frame_idx","t_frame_ns"])
        lw.writerow(["t_frame_ns"] + [f"theta_{i+1}" for i in range(N_ANGLES)] + ["quality"])

        # VideoWriter Initialisieren (Mit Größe des entzerrten Frames!)
        h_rec, w_rec = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(RUN_DIR/"video.mp4"), fourcc, FPS, (w_rec, h_rec))

        print("[Main] Aufnahme läuft...")
        frame_idx = 0
        
        while not stop_flag.is_set():
            ok, frame = cap.read()
            if not ok: break
            
            t_frame_ns = time.perf_counter_ns()
            fw.writerow([frame_idx, t_frame_ns])

            # --- ENTZERRUNG (Muss identisch zur Preview sein) ---
            if mtx is not None:
                frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
            # ----------------------------------------------------

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if use_new:
                corners, ids, rejected = detector.detectMarkers(gray)
            else:
                corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=params)

            thetas = [np.nan] * N_ANGLES
            quality = 0

            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)
                alpha = [None] * N_SEGMENTS
                
                for c, mid in zip(corners, ids.flatten()):
                    mid = int(mid)
                    if mid in ID_TO_SEGMENT:
                        alpha[ID_TO_SEGMENT[mid]] = marker_angle_from_corners(c[0])

                if all(a is not None for a in alpha):
                    ths = [math.degrees(wrap_pi(alpha[i+1] - alpha[i])) for i in range(N_ANGLES)]
                    thetas = ths
                    quality = 1

            # CSV schreiben
            thetas_csv = [f"{x:.4f}" if not math.isnan(x) else "nan" for x in thetas]
            lw.writerow([t_frame_ns, *thetas_csv, quality])

            # Video schreiben
            out.write(frame)

            # Anzeigen
            cv2.imshow(WIN, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            
            frame_idx += 1

    # Cleanup
    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()
    stop_flag.set()
    th.join()
    
    # Meta speichern
    (RUN_DIR/"meta.yaml").write_text(
        f"run: {RUN_NAME}\n"
        f"calibrated: {mtx is not None}\n"
        f"calib_file: {CALIB_FILE}\n"
        f"video_res: {w_rec}x{h_rec}@{FPS:.2f}\n"
        f"labels: theta_unit=deg\n", encoding="utf-8"
    )
    
    print(f"Fertig! Gespeichert in {RUN_DIR}")

if __name__ == "__main__":
    main()