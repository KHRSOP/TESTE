import pandas as pd
from collections import Counter
df = pd.read_csv('buscas.csv')

X_df = df [['home','busca','logado']]
Y_df = df ['comprou']

Xdummies_df = pd.get_dummies(X_df).astype(int)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_de_treino = 0.8
porcentagem_de_teste = 0.1

tamanho_de_treino = int(porcentagem_de_treino * len(Y))#len = comprimento 
tamanho_de_teste = porcentagem_de_teste * len(Y)
tamanho_de_validacao = len(Y) - tamanho_de_treino - tamanho_de_teste

# 0 ate 799
treino_dados = X[0:tamanho_de_treino]
treino_marcacoes = Y[0:tamanho_de_treino]

#800 ate 899
fim_de_teste = tamanho_de_treino + tamanho_de_teste

teste_dados = X[int(tamanho_de_treino):int(fim_de_teste)]
teste_marcacoes = Y[int(tamanho_de_treino):int(fim_de_teste)]

#900 ate 999
validacao_dados = X[int(fim_de_teste):]
validacao_marcacoes = Y[int(fim_de_teste):]


def fit_and_predict(nome,modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
     modelo.fit(treino_dados, treino_marcacoes)
     
     resultado = modelo.predict(teste_dados)
     
     acertos = (resultado == teste_marcacoes)
     
     total_de_acertos = sum(acertos)#soma
     total_de_elementos = len(teste_dados)
     
     taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
     
     msg = "Taxa de acerto do algoritimo {0}: {1}".format(nome, taxa_de_acerto)
     
     print(msg)
     return taxa_de_acerto
 
     
from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
    
from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict ("AdaBoostClassifier" ,modeloAdaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

acerto_base = max(Counter(validacao_marcacoes).itervalues())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
    
    
print("Taxa de acerto base: %f" % taxa_de_acerto_base)

total_de_elementos = len(teste_dados)
print("Total de teste: %d" % total_de_elementos)

if resultadoMultinomial > resultadoAdaBoost :
    vencedor = modeloMultinomial
else:
    vencedor = modeloAdaBoost

resultado = vencedor.predict(validacao_dados)
acertos = (resultado == validacao_marcacoes)
     
total_de_acertos = sum(acertos)#soma
total_de_elementos = len(validacao_marcacoes)
 
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
 
msg = "Taxa de acerto do vencedor dos algoritimos no mundo real {0}".format(taxa_de_acerto)
 
print(msg)


#OLD

#ACERTOS DO MODO ANTIGO
#diferencas = resultado - teste_marcacoes
#acertos = [d for d in diferencas if d == 0]
#total_de_acertos = len(acertos)
#total_de_elementos = len(teste_dados)
#taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

#a eficeiencia do algoritimo que chuta tudo 0 ou 1 // sum(Y) #soma dos Y 
#acerto_de_um = len(Y[Y==1])
#acerto_de_zero = len(Y[Y==0])
#taxa_de_acerto_base = 100.0 * max(acerto_de_um,acerto_de_zero)/ len(Y) #max = qual o maior dos dois

#print("Taxa de acerto base: %f" % taxa_de_acerto_base)