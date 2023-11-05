# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from flask import Flask, request, jsonify, render_template


# Load the data
df = pd.read_csv(
    "D:\\Jem\\Tugas Kuliah\\Data Mining\\UTS\\program\\penjualan_gitar.csv")

# Preprocess the data: One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=["Merek Gitar", "Tipe Gitar", "Warna"])

# Split the data into training and testing sets (80% training, 20% testing)
X = df_encoded.drop("Harga (dalam jutaan)", axis=1)
y = df_encoded["Harga (dalam jutaan)"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(X.columns.tolist())

# Train a linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluate the model
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")

# Save the trained model to a file
model_path = "D:\\Jem\\Tugas Kuliah\\Data Mining\\UTS\\program\\gitar_price_estimator.pkl"
joblib.dump(lr, model_path)

# Flask API
app = Flask(__name__)

# Load the trained model
model = joblib.load(model_path)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Ambil nilai dari formulir
        merek = request.form['merek']
        tipe = request.form['tipe']
        warna = request.form['warna']

        # Konversi ke format fitur
        data = {
            'Merek Gitar_Epiphone': [1 if merek == 'Epiphone' else 0],
            'Merek Gitar_Fender': [1 if merek == 'Fender' else 0],
            'Merek Gitar_Gibson': [1 if merek == 'Gibson' else 0],
            'Merek Gitar_Ibanez': [1 if merek == 'Ibanez' else 0],
            'Merek Gitar_Yamaha': [1 if merek == 'Yamaha' else 0],
            'Tipe Gitar_Akustik': [1 if tipe == 'Akustik' else 0],
            'Tipe Gitar_Bass': [1 if tipe == 'Bass' else 0],
            'Tipe Gitar_Elektrik': [1 if tipe == 'Elektrik' else 0],
            'Warna_Biru': [1 if warna == 'Biru' else 0],
            'Warna_Coklat': [1 if warna == 'Coklat' else 0],
            'Warna_Hitam': [1 if warna == 'Hitam' else 0],
            'Warna_Merah': [1 if warna == 'Merah' else 0],
            'Warna_Putih': [1 if warna == 'Putih' else 0]
        }

        # Buat DataFrame dari data
        df_predict = pd.DataFrame(data)

        # Lakukan prediksi
        prediction = model.predict(df_predict)[0]

        # Render template dengan prediksi
        return render_template('index.html', prediction=prediction, merek=merek, tipe=tipe, warna=warna)

    # Render template tanpa prediksi untuk permintaan GET
    return render_template('index.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
