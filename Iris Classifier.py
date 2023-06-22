from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Ładowanie zbioru danych Iris
iris_dataset = load_iris()

# Podział danych na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# Tworzenie DataFrame z danymi treningowymi
# Etykiety kolumn są oparte na iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# Tworzenie macierzy rozproszenia na podstawie DataFrame
# Kolorowanie punktów na podstawie y_train
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=0.8)

# Inicjalizacja klasyfikatora KNeighborsClassifier z 1 sąsiadem
knn = KNeighborsClassifier(n_neighbors=1)

# Trenowanie klasyfikatora na danych treningowych
knn.fit(X_train, y_train)

# Przykładowe dane do przewidzenia
X_new = np.array([[5, 2.9, 1, 0.2]])

# Przewidywanie klasy dla nowych danych
prediction = knn.predict(X_new)
print(f"\nPrognoza: {prediction}")
print(f"Typ kwiatu: {iris_dataset['target_names'][prediction]}")

# Przewidywanie klas dla danych testowych
y_pred = knn.predict(X_test)

# Obliczanie dokładności klasyfikatora na danych testowych
accuracy = knn.score(X_test, y_test)

# Wyświetlanie wyniku dokładności
print(f"\nWynik dla zestawu danych: {accuracy}\n\n")

# Wyświetlanie wykresu macierzy rozproszenia
# plt.show()
