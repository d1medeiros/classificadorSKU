#!-*- coding: utf8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier


def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes):
    k = 10
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv=k)
    taxa_de_acerto = np.mean(scores)
    msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
    print(msg)
    return taxa_de_acerto


def teste_real(modelo, validacao_dados, validacao_marcacoes):
    resultado = modelo.predict(validacao_dados)
    # acertos = resultado == validacao_marcacoes
    # total_de_acertos = sum(acertos)
    # total_de_elementos = len(validacao_marcacoes)
    # taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
    # msg = "Taxa de acerto do vencedor entre os dois algoritmos no mundo real: {0}".format(taxa_de_acerto)
    # print(msg)


def split_text(textosPuros):
    return textosPuros.str.lower().str.split(' ')


def vetorizar_texto_teste(texto, tradutor):
    vetor = [0] * len(tradutor)
    print "vetor->", vetor
    print "texto->", texto
    print "tradutor->", tradutor
    for palavra in texto:
        print palavra
        if palavra in tradutor:
            print palavra
            posicao = tradutor[palavra]
            vetor[posicao] += 1

    return vetor


def vetorizar_texto(texto, tradutor):
    vetor = [0] * len(tradutor)
    for palavra in texto:
        if palavra in tradutor:
            posicao = tradutor[palavra]
            vetor[posicao] += 1

    return vetor


def gera_tradutor(textosQuebrados):
    dicionario = set()
    for lista in textosQuebrados:
        dicionario.update(lista)

    total_de_palavras = len(dicionario)
    tuplas = zip(dicionario, range(total_de_palavras))
    return {palavra: indice for palavra, indice in tuplas}


def adaboost(x, y):
    porcentagem_de_treino = 0.8
    tamanho_de_treino = int(porcentagem_de_treino * len(y))

    treino_dados = x[0:tamanho_de_treino]
    treino_marcacoes = y[0:tamanho_de_treino]

    modelo_ada_boost = AdaBoostClassifier(random_state=0)
    modelo_ada_boost.fit(treino_dados, treino_marcacoes)
    return modelo_ada_boost


csv = 'file2.csv'
classificacoes = pd.read_csv(csv)
textosPuros = classificacoes['marca']
textosQuebrados = split_text(textosPuros)
tradutor = gera_tradutor(textosQuebrados)

X = [vetorizar_texto(texto, tradutor) for texto in textosQuebrados]
Y = classificacoes['classificacao']

modeloAdaBoost = adaboost(X, Y)

texto1 = "Smart TV LED 39 Semp"
texto2 = "Notebook Dell Inspiron I15-3567-A10P Intel Core 6ª i3 4GB 1TB "
texto3 = "Micro-ondas XXR34"

misterio = texto1
valor = [vetorizar_texto(misterio.lower().split(" "), tradutor)]


print modeloAdaBoost.predict(valor)
print misterio

