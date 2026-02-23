
# Spirob Angle Solving

## UV setup

Install dependencies from pyproject.toml and run via uv:

```bash
uv sync
```

Run a script using the uv environment:

```bash
uv run Full_Spirob_new_pcb/Sensor_Visualizer.py
```

## Workflow (Kurz)

1. Sensoren pruefen: Sensor_Visualizer starten und sicherstellen, dass alle
	Sensoren erkannt werden.
2. Winkel live ermitteln: linear_live_test starten.

## Sensor_Visualizer

Starts a live visualization of accelerometer and magnetometer data from the serial stream.
It expects binary frames from the MCU and plots accX/accY/accZ and magX/magY/magZ
per sensor ID in real time. If PyQtGraph is available it uses a fast GUI; otherwise
it falls back to a slower Matplotlib view. There is also a demo mode if the serial
port cannot be opened.

Start it:

```bash
uv run Full_Spirob_new_pcb/Sensor_Visualizer.py
```

## linear_live_test

Loads a linear calibration file (linear_calib.json) and estimates joint angles in
real time using the linear model:

	angle = slope * mag_value + intercept

It supports two modes:
- Live serial mode with optional PyQtGraph plotting
- Offline replay from a sensors.csv file

Start it:

```bash
uv run Full_Spirob_new_pcb/linear_live_test.py
```

