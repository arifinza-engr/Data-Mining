# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from flask import Flask, request, jsonify, render_template


# Load the data
df = pd.read_csv(
    "D:\\Jem\\Tugas Kuliah\\Data Mining\\UTS\\program\\guitar.csv")

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
    # Definisikan feature_columns di sini
    feature_columns = X.columns.tolist()

    if request.method == 'POST':
        # Ambil nilai dari formulir
        merek = request.form['merek']
        tipe = request.form['tipe']
        warna = request.form['warna']

        # Konversi ke format fitur
        data = {
            'Merek Gitar_Harley Benton': [1 if merek == 'Harley Benton' else 0],
            'Merek Gitar_Gibson': [1 if merek == 'Gibson' else 0],
            'Merek Gitar_Gretsch': [1 if merek == 'Gretsch' else 0],
            'Merek Gitar_Epiphone': [1 if merek == 'Epiphone' else 0],
            'Merek Gitar_DAngelico': [1 if merek == 'DAngelico' else 0],
            'Merek Gitar_ESP': [1 if merek == 'ESP' else 0],
            'Merek Gitar_Maybach': [1 if merek == 'Maybach' else 0],
            'Merek Gitar_Duesenberg': [1 if merek == 'Duesenberg' else 0],
            'Merek Gitar_Solar Guitars': [1 if merek == 'Solar Guitars' else 0],
            'Merek Gitar_Larry Carlton': [1 if merek == 'Larry Carlton' else 0],
            'Merek Gitar_Godin': [1 if merek == 'Godin' else 0],
            'Merek Gitar_PRS': [1 if merek == 'PRS' else 0],
            'Merek Gitar_Cort': [1 if merek == 'Cort' else 0],
            'Merek Gitar_Hagstrom': [1 if merek == 'Hagstrom' else 0],
            'Merek Gitar_Chapman Guitars': [1 if merek == 'Chapman Guitars' else 0],
            'Merek Gitar_Kramer Guitars': [1 if merek == 'Kramer Guitars' else 0],
            'Merek Gitar_Taylor': [1 if merek == 'Taylor' else 0],
            'Merek Gitar_Heritage Guitar': [1 if merek == 'Heritage Guitar' else 0],
            'Merek Gitar_Schecter': [1 if merek == 'Schecter' else 0],
            'Merek Gitar_Stanford': [1 if merek == 'Stanford' else 0],
            'Merek Gitar_Jackson': [1 if merek == 'Jackson' else 0],
            'Merek Gitar_Guild': [1 if merek == 'Guild' else 0],
            'Merek Gitar_FGN': [1 if merek == 'FGN' else 0],
            'Merek Gitar_Framus': [1 if merek == 'Framus' else 0],
            'Merek Gitar_Journey Instruments': [1 if merek == 'Journey Instruments' else 0],
            'Merek Gitar_Danelectro': [1 if merek == 'Danelectro' else 0],

            'Tipe Gitar_Akustik': [1 if tipe == 'Akustik' else 0],
            'Tipe Gitar_Elektrik': [1 if tipe == 'Elektrik' else 0],

            'Warna_natural': [1 if warna == 'natural' else 0],
            'Warna_silver': [1 if warna == 'silver' else 0],
            'Warna_putih': [1 if warna == 'putih' else 0],
            'Warna_biru': [1 if warna == 'biru' else 0],
            'Warna_kuning': [1 if warna == 'kuning' else 0],
            'Warna_gold': [1 if warna == 'gold' else 0],
            'Warna_cherry': [1 if warna == 'cherry' else 0],
            'Warna_sunburst': [1 if warna == 'sunburst' else 0],
            'Warna_cokelat': [1 if warna == 'cokelat' else 0],
            'Warna_merah': [1 if warna == 'merah' else 0],
            'Warna_tobacco sunburst': [1 if warna == 'tobacco sunburst' else 0],
            'Warna_sun yellow': [1 if warna == 'sun yellow' else 0],
            'Warna_hijau': [1 if warna == 'hijau' else 0],
            'Warna_pink': [1 if warna == 'pink' else 0],
            'Warna_ungu': [1 if warna == 'ungu' else 0],
            'Warna_orange': [1 if warna == 'orange' else 0],
            'Warna_midnight blue': [1 if warna == 'midnight blue' else 0],
            'Warna_olive drab': [1 if warna == 'olive drab' else 0],
            'Warna_hitam': [1 if warna == 'hitam' else 0]
        }

        # Buat DataFrame dari data
        df_predict = pd.DataFrame(data)

        # Mengatur ulang kolom dataframe untuk memastikan urutan yang benar
        df_predict = df_predict.reindex(columns=feature_columns, fill_value=0)

        # Lakukan prediksi
        prediction = model.predict(df_predict)[0]

        # Render template dengan prediksi
        return render_template('index.html', prediction=prediction, merek=merek, tipe=tipe, warna=warna)

    # Render template tanpa prediksi untuk permintaan GET
    return render_template('index.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=False)
