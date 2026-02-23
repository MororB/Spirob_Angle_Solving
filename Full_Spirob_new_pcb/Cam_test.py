import cv2

def find_available_cameras(limit=10):
    available_cameras = []
    
    print("Suche nach verfügbaren Kameras...")
    print("Drücke 'q' oder 'ESC' beim jeweiligen Fenster, um die nächste Kamera zu prüfen.")

    for i in range(limit):
        # CAP_DSHOW wird unter Windows empfohlen, um IDs schneller zu finden
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                window_name = f"Kamera Index: {i}"
                cv2.imshow(window_name, frame)
                
                print(f"[ERFOLG] Kamera mit Index {i} gefunden.")
                
                # Warte auf Tastendruck, um zum nächsten Index zu springen
                key = cv2.waitKey(0)
                cv2.destroyWindow(window_name)
                
                if key == ord('q') or key == 27:
                    break
            cap.release()
        else:
            print(f"[INFO] Index {i}: Keine Kamera.")

    print("-" * 30)
    print(f"Gefundene Indizes: {available_cameras}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    find_available_cameras()