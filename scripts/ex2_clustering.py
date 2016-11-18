
# coding: utf-8

# Exemplo 2
# =========
#
# Voltaremos agora aos dados de dígitos, mas agora com métodos de
# aprendizado não supervisionado. Vamos aplicar um algoritmo denominado
# *k-means*, que é capaz de agrupar os dados em k grupos dada a proximidade
# dos dados.

# In[ ]:

from sklearn import datasets
from sklearn.cluster import KMeans


digits = datasets.load_digits()

kmeans.fit(digits.data)


# Exercício 2.1: visualizando as classes
# --------------------------------------
#
# Tente explicar por que isto não funciona? Rode algumas vezes até que
# algum dos números de uma alta taxa de acerto.

# In[ ]:

from sklearn.metrics import classification_report

kmeans.fit(digits.data)
predicted = kmeans.predict(digits.data)

print(classification_report(expected, predicted))


# Mapeando as classes
# -------------------
#
# Vamos tentar encontrar uma correspondência entre os grupos
# identificados via *clustering* e os dígitos. O que isto significa?
# Isto contradiz a ideia de um método não supervisionado?

# In[ ]:

print(kmeans.labels_[10:20])
print(digits.target[10:20])

mapping = dict(zip(kmeans.labels_[10:20], digits.target[10:20]))
print mapping

labels = [mapping[lbl] for lbl in kmeans.labels_]

print(classification_report(digits.target, labels))


# Desafio
# -------
#
# Utilizar uma SVM com apenas 50 imagens e montar um algoritmo semi-supervisionado.

# In[ ]:

from sklearn.metrics import classification_report
svm = SVC(kernel='linear')

# Treinamento com 50 imagens
svm.fit(digits.data[:50], digits.target[:50])

