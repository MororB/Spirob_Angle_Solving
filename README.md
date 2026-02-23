
# Spirob Angle Solving

## UV Setup

Abhaengigkeiten aus pyproject.toml installieren und via uv laufen lassen:

```bash
uv sync
```

Skript mit dem uv-Environment starten:

```bash
uv run Full_Spirob_new_pcb/Sensor_Visualizer.py
```

## Ablauf (Kurz)

1. Sensoren pruefen: Sensor_Visualizer starten und sicherstellen, dass alle
   Sensoren erkannt werden.
2. Winkel live ermitteln: linear_live_test starten.

## Sensor_Visualizer

Startet eine Live-Visualisierung der Beschleunigungs- und Magnetfeld-Daten vom
seriellen Stream. Erwartet binaere Frames vom MCU und plottet accX/accY/accZ
und magX/magY/magZ pro Sensor-ID in Echtzeit. Falls PyQtGraph verfuegbar ist,
wird eine schnelle GUI genutzt, sonst faellt es auf eine langsamere Matplotlib-
Ansicht zurueck. Es gibt auch einen Demo-Modus, wenn der serielle Port nicht
geoeffnet werden kann.

Start:

```bash
uv run Full_Spirob_new_pcb/Sensor_Visualizer.py
```

## linear_live_test

Laedt eine lineare Kalibrierdatei (linear_calib.json) und schaetzt Gelenkwinkel
in Echtzeit mit dem linearen Modell:

    angle = slope * mag_value + intercept

Unterstuetzt zwei Modi:
- Live-Seriell mit optionaler PyQtGraph-Plotanzeige
- Offline-Wiedergabe aus einer sensors.csv

Start:

```bash
uv run Full_Spirob_new_pcb/linear_live_test.py
```

