import numpy as np
import pandas as pd

# 1. Membaca dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# 2. Encoding data kategori (Atribut)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Dapatkan posisi kolom numerik setelah encoding
# Misalnya, jika Anda memiliki 'age' dan 'salary' sebagai kolom numerik
num_cols_start = X.shape[1] - 2

# 3. Menghilangkan missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, num_cols_start:] = imputer.fit_transform(X[:, num_cols_start:])

# 4. Encoding data kategori (Class / Label)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Membagi dataset ke dalam training set dan test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, num_cols_start:] = sc.fit_transform(X_train[:, num_cols_start:])
X_test[:, num_cols_start:] = sc.transform(X_test[:, num_cols_start:])  # Gunakan transform saja di X_test

print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("y_train")
print(y_train)
print("y_test")
print(y_test)
