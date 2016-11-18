
# coding: utf-8

# Exemplo 1
# =========
#
#  Neste exemplo usaremos um conjunto de dados de exemplo chamado *digits*, composta por 1797 imagens de dígitos de tamanho 8x8 digitalizadas e preprocessadas, disponível na biblioteca [scikit-learn](http://scikit-learn.org/). Para saber mais sobre os dados consulte [a documentação](http://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html).

# In[ ]:

from sklearn import datasets

digits = datasets.load_digits()
print("Numero de imagens:", len(digits.images))


# Visualizando os dados
# ---------------------
#
# Para termos uma ideia do que estamos trabalhando, é sempre bom fazer uma análise exploratória
# dos dados **antes** de aplicar algoritmos de *machine learning*. Para termos uma ideia de como são
# os dados veja o exemplo abaixo.

# In[ ]:

import matplotlib.pyplot as plt

# Imprime a figura de exemplo
plt.imshow(digits.images[0], cmap=plt.cm.binary)
plt.show()
# Imprime a saída correspondente à figura de exemplo
print("Class:", digits.target[0])


# Aplicando um algorítmo de ML
# ----------------------------
#
# Finalmente, vamos ao nosso primeiro exemplo. Usaremos um classificador chamado
# **SVM** (*Support Vector Machine*) para tentar montar um modelo que dado um dígito
# escrito a mão na entrada, preveja qual é número correspondente

# In[ ]:

# Modelo de classificação com SVM
from sklearn.svm import SVC
from sklearn.metrics import classification_report

svm = SVC(kernel='linear')

x_train = digits.data[:1000]
y_train = digits.target[:1000]

# Treinamento (com 1000 imagens)
svm.fit(x_train, y_train)

# Testes (com 797 imagens)
x_test = digits.data[1000:]
y_predicted = svm.predict(x_test)
y_expected = digits.target[1000:]

print(classification_report(y_expected, y_predicted))


# O que é uma SVM?
# ----------------
#
# Por ora não precisamos saber detalhes, mas uma explicação rápida é a seguinte:
# Vamos imaginar um conjunto  de dados com apenas duas dimensoes de entrada (x1 e x2) e com duas classes
# (azul e vermelha) como o da figura abaixo.
# Uma SVM é capaz de encontrar uma reta que separa o conjunto com a maior distância
# média possível entre os pontos da fronteira das duas classes.
#
# ![Sets linearmente separáveis](img/svm.png)
#
# Isto também funciona com um número maior de entradas, como no nosso exemplo, em que temos
# 16x16=256 entradas, mas a separação deixa de ser um plano e passa a ser o que chamamos de
# hiperplano em 255 dimensões.
#
# Para mais detalhes, consulte:
#
# Obs: está e uma explicação para uma SVM com kernel linear, como a que usamos neste
# exemplo. Outros tipos existem e a diferença básica entre eles é que ao invés de
# um hiperplano, outros modelos geram estruturas geométricas diferentes, que podem
# ser mais úteis em certos problemas.

# Exercício 1.1: Random Forests
# -----------------------------
#
# Reproduza o primeiro exemplo usando o classificador por *random forests*,
# você pode variar o número de árvores de decisão usadas pelo parâmetro `n_estimators`
# e tentar melhorar a precisão do algorítmo.

# In[ ]:

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10)


# O que é uma Random Forest?
# --------------------------
#
# Resumindo, um algoritmo de *Random Forest* divide os dados em N grupos e monta N
# árvores de decisão (como a da imagem abaixo) para cada grupo.
# Ao final, o resultado das várias árvores é combinado para gerar a resposta final.
# Note que a motivação teórica deste algoritmo é bem diferente de uma SVM, mas
# as duas tem papeis e resultados parecidos.
#
# ![Árvore de decisão](img/tree.png)

# A informação está nos dados
# ---------------------------
#
# Menos dados pode significar uma precisão pior, já que a informação do
# nosso modelo está nos dados, e não diretamente no algoritmo empregado.
# Mas isso nem sempre é verdade, pois o nosso algoritmo pode ter limitações.

# In[ ]:

svm = SVC(kernel='linear')

# Treinamento com 50 imagens
svm.fit(digits.data[:50], digits.target[:50])

# Testes
predicted = svm.predict(digits.data[1000:])
expected = digits.target[1000:]
print(classification_report(expected, predicted))


# Exercício 1.2: Rede Neural Perceptron
# -------------------------------------
#
# Vamos agora repetir o problema para uma rede neural do tipo *Perceptron*.
# Também não vamos entrar em detalhes do algoritmo agora, mas podemos dizer
# que ele é um algoritmo mais versátil que SVMs e *Random Forests*.

# In[ ]:

# Obs: não executa no binder
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(100))


# Exercício 1.3: Spam
# -------------------
#
# Agora é hora de aplicar os algorítimos "de verdade".
# Vamos ler um conjunto de dados de mensagens SMS entre
# mensagens reais e *spams* e devemos classificá-las corretamente.
#
# Use o algorítmo que quiser, para facilitar a vida já fizemos parte do préprocessamento.

# In[ ]:

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

# Abre o dataset
data = open('../data/spam/SMSSpamCollection').read()

# Cada linha é uma mensagem e a ultima esstá em branco
data = data.split('\n')[:-1]
# as classes estão antes do primeiro espaço
classes = [line.split('\t', 1)[0] for line in data[:-1]]
# As mensagens estão depois do primeiro espaço
messages = [line.split('\t', 1)[1] for line in data[:-1]]

# Transforma as mensagens em uma grande matriz (esparsa) de palavras
count_vect = CountVectorizer()
features = count_vect.fit_transform(messages)

##### Aqui é por sua conta #####


# Desafio: Melhorar a precisão dos exemplos
# -----------------------------------------
#
# Não discutimos isto, mas os algoritmos utilizados tem parâmetros que permitem
# obter um modelo melhor. Busque na documentação os algoritmos utilizados
# (ou outros se preferir) e procure melhorar a precisão.
