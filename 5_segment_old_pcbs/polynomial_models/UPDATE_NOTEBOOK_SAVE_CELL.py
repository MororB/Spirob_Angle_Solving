# ===============================================================
# ZUSÄTZLICHE ZELLE FÜR polynomial.ipynb - Abschnitt 8
# Ersetze die alte Speicher-Zelle mit diesem Code:
# ===============================================================

# Bestes Modell speichern
import os

# Ordner erstellen falls nicht vorhanden
save_dir = f'polynomial_models/polynomial_degree_{int(best_degree)}'
os.makedirs(save_dir, exist_ok=True)

# Dateinamen
model_filename = os.path.join(save_dir, 'model.pkl')
poly_filename = os.path.join(save_dir, 'poly_features.pkl')
scaler_filename = os.path.join(save_dir, 'scaler.pkl')

# Speichern
joblib.dump(best_model_data['model'], model_filename)
joblib.dump(best_model_data['poly'], poly_filename)
joblib.dump(best_model_data['scaler'], scaler_filename)

print(f"Modell gespeichert in: {save_dir}")
print(f"  - model.pkl")
print(f"  - poly_features.pkl")
print(f"  - scaler.pkl")

# Info für Live-Test ausgeben
print(f"\n{'='*50}")
print(f"Für Live-Test: Setze MODEL_FOLDER_NAME = 'polynomial_degree_{int(best_degree)}'")
print(f"in live_test_polynomial.py")
print(f"{'='*50}")
