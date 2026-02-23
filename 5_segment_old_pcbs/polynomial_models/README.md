# Polynomial Models Directory

Dieser Ordner enthält die trainierten **Polynomial Regression** Modelle für die Spirob Winkelerkennung.

## Struktur

Jedes trainierte Modell wird in einem eigenen Unterordner gespeichert:

```
polynomial_models/
├── polynomial_degree_2/
│   ├── model.pkl          # Ridge Regression Modell
│   ├── poly_features.pkl  # PolynomialFeatures Transformer
│   └── scaler.pkl         # StandardScaler
│
├── polynomial_degree_3/
│   ├── model.pkl
│   ├── poly_features.pkl
│   └── scaler.pkl
│
└── UPDATE_NOTEBOOK_SAVE_CELL.py  # Code für Notebook-Update
```

## Dateien pro Modell

### `model.pkl`
Das trainierte Ridge Regression Modell.
- **Typ**: `sklearn.linear_model.Ridge`
- **Enthält**: Koeffizienten für die Winkelberechnung
- **Größe**: ~7 KB (Grad 3) 

### `poly_features.pkl`
Der Polynomial Features Transformer.
- **Typ**: `sklearn.preprocessing.PolynomialFeatures`
- **Funktion**: Erstellt polynomiale Features aus Rohdaten
- **Beispiel**: Aus [x, y, z] wird [x, y, z, x², xy, xz, y², yz, z², ...]

### `scaler.pkl`
Der StandardScaler für Feature-Normalisierung.
- **Typ**: `sklearn.preprocessing.StandardScaler`
- **Funktion**: Normalisiert Features auf Mittelwert=0, Standardabweichung=1

## Verwendung im Live-Test

### 1. Modell trainieren und speichern

Im Jupyter Notebook `polynomial.ipynb`:
- Führe alle Zellen aus
- **WICHTIG**: Ersetze die Speicher-Zelle (Abschnitt 8) mit dem Code aus `UPDATE_NOTEBOOK_SAVE_CELL.py`
- Das Modell wird dann automatisch in den richtigen Ordner gespeichert

### 2. Live-Test starten

```bash
python live_test_polynomial.py
```

**Konfiguration** in `live_test_polynomial.py`:
```python
SERIAL_PORT = "COM5"  # Deinen COM-Port eintragen
MODEL_FOLDER_NAME = "polynomial_degree_3"  # Ordnername hier eintragen
```

## Vorteile gegenüber LSTM

| Aspekt | Polynomial | LSTM |
|--------|-----------|------|
| **Inferenzzeit** | ~0.006 ms | ~2 ms (300x langsamer) |
| **Modellgröße** | ~7 KB | ~400 KB |
| **Buffer nötig?** | ❌ Nein | ✅ Ja (10 Samples) |
| **Deployment** | Einfach | Komplex |
| **Mikrocontroller** | ✅ Ja | ⚠️ Schwierig |

## Benchmark-Ergebnisse (Grad 3)

```
Test MAE:  1.80°  (Durchschnittsfehler)
Test MSE:  7.33   
Test R²:   0.939  (93.9% Varianz erklärt)
Inferenz:  0.006 ms pro Sample
Features:  454 polynomiale Features
```

## Troubleshooting

### Modell nicht gefunden
```
FileNotFoundError: Modell nicht gefunden
```
**Lösung**: 
1. Überprüfe `MODEL_FOLDER_NAME` in `live_test_polynomial.py`
2. Stelle sicher, dass das Notebook die Modelle gespeichert hat
3. Überprüfe, dass die Ordnerstruktur stimmt

### Falsche Input-Dimension
```
ValueError: X has 12 features, but model is expecting 454
```
**Lösung**: 
- Das Modell wurde mit anderen Sensoren trainiert
- Überprüfe `ACTIVE_SENSOR_IDS` im Training

### Serial Port Error
```
Serial Error: could not open port 'COM5'
```
**Lösung**:
- Überprüfe den COM-Port im Geräte-Manager
- Stelle sicher, dass keine andere Anwendung den Port nutzt

## Nächste Schritte

1. ✅ Notebook ausführen und Modell speichern
2. ✅ `live_test_polynomial.py` konfigurieren  
3. ✅ Live-Test durchführen
4. 📊 Ergebnisse mit LSTM vergleichen
5. 🚀 Auf Mikrocontroller deployen

## Support

Bei Problemen:
- Überprüfe die Notebook-Ausgabe
- Vergleiche mit `live_test.py` (LSTM) für Referenz
- Stelle sicher, dass alle `.pkl` Dateien vorhanden sind
