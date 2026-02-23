import csv, time, threading, struct
import serial

HEADER = b"\xAA\x55"
NUM_CHIPS = 5
TXYZ_SIZE = 16               # 4 floats (t,x,y,z) à 4 Byte
BOARD_PAYLOAD = NUM_CHIPS * TXYZ_SIZE

def read_exact(ser, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            raise RuntimeError("Serial timeout while reading payload")
        buf.extend(chunk)
    return bytes(buf)

def next_board_frame(ser):
    # Header suchen: 0xAA 0x55
    while True:
        b = ser.read(1)
        if not b:
            return None
        if b == HEADER[:1]:
            b2 = ser.read(1)
            if b2 == HEADER[1:]:
                break

    board_id_b = ser.read(1)
    if not board_id_b:
        return None
    board_id = board_id_b[0]

    payload = read_exact(ser, BOARD_PAYLOAD)

    sensors = []
    for i in range(NUM_CHIPS):
        chunk = payload[i*TXYZ_SIZE:(i+1)*TXYZ_SIZE]
        t_s, x, y, z = struct.unpack("<ffff", chunk)
        sensors.append((i, t_s, x, y, z))
    return board_id, sensors

def serial_logger(port, baud, stop_flag, out_csv):
    frames = 0
    rows = 0
    try:
        ser = serial.Serial(port, baud, timeout=0.2)
        ser.reset_input_buffer()

        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_ns","board_id","sensor_id","t","x","y","z"])
            f.flush()

            last_flush = time.time()
            last_print = time.time()

            while not stop_flag.is_set():
                frm = next_board_frame(ser)
                if frm is None:
                    continue

                board_id, sensors = frm
                t_ns = time.perf_counter_ns()

                for sensor_id, t_s, x, y, z in sensors:
                    w.writerow([t_ns, board_id, sensor_id, t_s, x, y, z])
                    rows += 1

                frames += 1

                # alle ~0.5s flushen, damit Datei live wächst
                if time.time() - last_flush > 0.5:
                    f.flush()
                    last_flush = time.time()

                # alle ~1s Status ausgeben
                if time.time() - last_print > 1.0:
                    print(f"[serial] frames={frames} rows={rows}")
                    last_print = time.time()

        ser.close()

    except Exception as e:
        print("[serial] ERROR:", repr(e))

# --- Start/Stop Beispiel ---
stop = threading.Event()
t = threading.Thread(target=serial_logger, args=("COM6", 1_000_000, stop, "sensors.csv"))
t.start()

print("Läuft... 5 Sekunden")
time.sleep(5)

stop.set()
t.join()
print("Fertig.")
