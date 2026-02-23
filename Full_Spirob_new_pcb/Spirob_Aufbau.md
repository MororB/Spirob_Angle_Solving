# Spirob – Physischer Aufbau

## Überblick

Der Spirob ist ein Roboterarm bestehend aus **14 Segmenten**, die nach oben hin
kleiner werden (fraktaler Baum). Zwischen je zwei Segmenten sitzt ein **Gelenk (Theta)**.
Die Winkelmessung erfolgt über **6 PCBs mit 3D-Hall-Sensoren** (MLX90393 + MMA8452Q),
die über **LTC4316 I²C-Adress-Translator** (XOR-IDs) angesprochen werden.

## Segmente, Sensoren und Gelenke

Das Muster wiederholt sich: **Sensor → Gelenk → Magnet → Gelenk → Sensor → ...**

```
┌──────────────────────────────────────────────────────────────────┐
│  Segment  │  Inhalt          │  Gelenk darüber   │  Anmerkung  │
├───────────┼──────────────────┼───────────────────-┼─────────────┤
│  Seg  1   │  SENSOR 1 (PCB)  │    Theta  1        │  unterstes  │
│  Seg  2   │  Magnet A        │    Theta  2        │             │
│  Seg  3   │  SENSOR 2 (PCB)  │    Theta  3        │             │
│  Seg  4   │  Magnet B        │    Theta  4        │             │
│  Seg  5   │  SENSOR 3 (PCB)  │    Theta  5        │             │
│  Seg  6   │  Magnet C        │    Theta  6        │             │
│  Seg  7   │  SENSOR 4 (PCB)  │    Theta  7        │             │
│  Seg  8   │  Magnet D        │    Theta  8        │             │
│  Seg  9   │  SENSOR 5 (PCB)  │    Theta  9        │             │
│  Seg 10   │  Magnet E        │    Theta 10        │             │
│  Seg 11   │  SENSOR 6 (PCB)  │    Theta 11        │  letzter    │
│  Seg 12   │  (leer)          │    Theta 12        │  zu klein   │
│  Seg 13   │  (leer)          │    Theta 13        │  zu klein   │
│  Seg 14   │  Magnet F        │    —               │  Spitze     │
└──────────────────────────────────────────────────────────────────┘
```

**Gesamt:** 14 Segmente, 13 Gelenke (Theta 1–13), 6 Sensor-PCBs, 6 Magnete

## Sensor → Theta Zuordnung (Messprinzip)

Jeder Hall-Sensor misst das Magnetfeld der **direkt benachbarten Magnete**.
Wenn sich ein Gelenk zwischen Sensor und Magnet dreht, ändert sich das
gemessene Magnetfeld – daraus lässt sich der Gelenkwinkel ableiten.

| Sensor (physisch) | Sensor-ID (Code) | Segment | Misst Theta(s)       | Erklärung                        |
|:-----------------:|:-----------------:|:-------:|:--------------------:|:---------------------------------|
|    Sensor 1       |        0          |    1    | **Theta 1**          | Nur Magnet A oben (kein Magnet unten) |
|    Sensor 2       |        1          |    3    | **Theta 2, Theta 3** | Magnet A unten + Magnet B oben   |
|    Sensor 3       |        2          |    5    | **Theta 4, Theta 5** | Magnet B unten + Magnet C oben   |
|    Sensor 4       |        3          |    7    | **Theta 6, Theta 7** | Magnet C unten + Magnet D oben   |
|    Sensor 5       |        4          |    9    | **Theta 8, Theta 9** | Magnet D unten + Magnet E oben   |
|    Sensor 6       |        5          |   11    | **Theta 10, Theta 11** | Magnet E unten + Magnet F (via leere Segmente) |

### Nicht messbar

- **Theta 12** und **Theta 13**: Keine Sensor-PCBs in Segmenten 12–13
  (zu kleine Segmente für Platinen).
- Der letzte Magnet (F) in Segment 14 hat keinen Sensor darüber.

## Hardware

- **Hall-Sensor:** MLX90393 (3-Achsen Magnetometer: magX, magY, magZ)
- **Beschleunigungssensor:** MMA8452Q (3-Achsen: accX, accY, accZ)
- **Adressierung:** LTC4316 I²C-Adress-Translator mit XOR-IDs  
  _(Die Sensor-ID im Datenstream = XOR-ID des LTC4316)_
- **Controller:** ESP32 (I²C @ 400kHz, Serial @ 1Mbit/s)
- **Framerate:** 50 Hz (20ms Intervall)

## Datenformat

Jeder Frame enthält pro Sensor:
```
sensor_id (uint8) + accX, accY, accZ, magX, magY, magZ (6 × float32)
```

## Hinweise zur Kalibrierung

- Nur die in der Tabelle definierten Sensor→Theta-Paare ergeben physikalisch
  sinnvolle Kalibrierungen.
- Alle anderen Kombinationen zeigen typischerweise **negative R²-Werte**
  (schlechter als ein konstantes Modell).
- Die lineare Kalibrierung nutzt Fixpunkte bei ±30° als Referenz.
