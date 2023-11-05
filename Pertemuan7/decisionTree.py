# Impor modul yang diperlukan
from sklearn import datasets, tree
from sklearn.tree import DecisionTreeClassifier
import pydotplus
import pandas as pd
import numpy as np

# Muat dataset Iris bawaan
iris = datasets.load_iris()
fitur_iris = iris['data']
target_iris = iris['target']

# Latih pohon keputusan untuk dataset Iris
pohon_iris = DecisionTreeClassifier(random_state=0)
model_iris = pohon_iris.fit(fitur_iris, target_iris)

# Visualisasi pohon keputusan Iris
data_dot = tree.export_graphviz(
    model_iris,
    out_file=None,
    feature_names=iris['feature_names'],
    class_names=iris['target_names']
)
grafik = pydotplus.graph_from_dot_data(data_dot)
grafik.set_graphviz_executables(
    {"dot": r"C:\Program Files (x86)\Graphviz\bin\dot.exe"})
grafik.write_png("iris.png")

# Muat dataset kustom dari lokasi tertentu
lokasi_dataset = 'C:\\Users\\ACER\\Documents\\TugasKuliah\\Data Mining\\Pertemuan7\\Dataset-Iris.csv'
dataset_kustom = pd.read_csv(lokasi_dataset, sep=';')
dataset_kustom["Species"] = pd.factorize(dataset_kustom.Species)[0]
dataset_kustom = dataset_kustom.drop(labels="Id", axis=1).to_numpy()

# Bagi data menjadi data latih dan data uji
data_latih = np.concatenate(
    (dataset_kustom[0:40, :], dataset_kustom[50:90, :]), axis=0)
data_uji = np.concatenate(
    (dataset_kustom[40:50, :], dataset_kustom[90:100, :]), axis=0)

# Pisahkan fitur dan target
fitur_latih = data_latih[:, 0:4]
target_latih = data_latih[:, 4]
fitur_uji = data_uji[:, 0:4]
target_uji = data_uji[:, 4]

# Latih pohon keputusan untuk dataset kustom
model_kustom = tree.DecisionTreeClassifier().fit(fitur_latih, target_latih)

# Prediksi label data uji
hasil_prediksi = model_kustom.predict(fitur_uji)

# Hitung akurasi prediksi
jumlah_benar = (hasil_prediksi == target_uji).sum()
jumlah_salah = (hasil_prediksi != target_uji).sum()
akurasi = jumlah_benar / (jumlah_benar + jumlah_salah) * 100

# Tampilkan hasil
print("Label sebenarnya:", target_uji)
print("Hasil prediksi:", hasil_prediksi)
print("Prediksi benar:", jumlah_benar, "data")
print("Prediksi salah:", jumlah_salah, "data")
print("Akurasi:", akurasi, "%")
