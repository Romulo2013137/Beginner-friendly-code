# Importação das bibliotecas necessárias
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

# Carregando o conjunto de dados de dígitos
digits = datasets.load_digits()

# Divisão dos dados em conjuntos de treino e teste
# 80% dos dados serão usados para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Definição do modelo
# Neste caso, estamos usando um Support Vector Machine (SVM) com um kernel radial basis function (RBF)
model = svm.SVC(kernel='rbf', gamma=0.001, C=100.)

# Treinamento do modelo com os dados de treino
model.fit(X_train, y_train)

# Previsão utilizando os dados de teste
predictions = model.predict(X_test)

# Avaliação do modelo
# A acurácia é uma métrica comum para avaliar modelos de classificação
print("Acurácia:", metrics.accuracy_score(y_test, predictions))

# Imprimindo a matriz de confusão
print("Matriz de Confusão:")
print(metrics.confusion_matrix(y_test, predictions))
