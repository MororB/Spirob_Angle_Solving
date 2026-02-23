import cv2
import time

# ---------------- Einstellungen ----------------
CAM_IDX = 0
 # ggf. 1/2 probieren
BACKENDS = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF",  cv2.CAP_MSMF),
]

REQUEST_W, REQUEST_H = 1920, 1080
REQUEST_FPS = 60
USE_MJPG = True   # viele USB cams liefern 1080p stabiler mit MJPG

WIN = "Camera Test (ESC=quit, B=backend switch)"

# ---------------- Helfer ----------------
def open_cam(backend):
    cap = cv2.VideoCapture(CAM_IDX, backend)

    # Wunsch-Format setzen (best effort)
    if USE_MJPG:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_H)
    cap.set(cv2.CAP_PROP_FPS, REQUEST_FPS)

    # Warmup
    time.sleep(0.2)
    return cap

def main():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    backend_i = 0
    backend_name, backend = BACKENDS[backend_i]
    cap = open_cam(backend)

    if not cap.isOpened():
        print("Kamera konnte nicht geöffnet werden.")
        return

    # FPS-Messung
    t0 = time.time()
    frames = 0
    fps_meas = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Kein Frame. (Backend/Index?)")
            break

        frames += 1
        if time.time() - t0 >= 1.0:
            fps_meas = frames / (time.time() - t0)
            t0 = time.time()
            frames = 0

        h, w = frame.shape[:2]

        # Werte, die OpenCV "glaubt" (nicht immer zuverlässig)
        w_req = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h_req = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps_req = cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc >> 8*i) & 0xFF) for i in range(4)])

        # Overlay-Text
        lines = [
            f"Backend: {backend_name} | CAM_IDX={CAM_IDX}",
            f"Requested: {REQUEST_W}x{REQUEST_H} @ {REQUEST_FPS}fps | MJPG={USE_MJPG}",
            f"CAP says: {w_req:.0f}x{h_req:.0f} @ {fps_req:.1f}fps | FOURCC={fourcc_str}",
            f"Actual frame: {w}x{h} | measured FPS: {fps_meas:.1f}",
            "Keys: ESC quit | B switch backend",
        ]

        y = 30
        for s in lines:
            cv2.putText(frame, s, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
            cv2.putText(frame, s, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
            y += 30

        # Fenstergröße holen + skalieren fürs Display (nur Anzeige!)
        x, y, ww, hh = cv2.getWindowImageRect(WIN)
        show = cv2.resize(frame, (ww, hh), interpolation=cv2.INTER_LINEAR) if (ww > 0 and hh > 0) else frame
        cv2.imshow(WIN, show)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key in (ord('b'), ord('B')):
            # Backend wechseln
            cap.release()
            backend_i = (backend_i + 1) % len(BACKENDS)
            backend_name, backend = BACKENDS[backend_i]
            cap = open_cam(backend)
            print("Switched to backend:", backend_name)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
