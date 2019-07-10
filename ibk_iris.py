# coding: utf-8
import csv
import vet
import random
import statistics as st

# Declaração de variávei global para posterior uso
base = []
p = 0
def abre_csv(filename):
    """
    Leitura do arquivo csv.
    :param filename: o nome do arquivo
    :return: uma lista com as linhas do arquivo (dataset)
    """
    with open(filename, 'r') as file:
        lines = csv.reader(file)
        return list(lines)

def calc(dataset):
    """
    Efetua os cálculos de mínimo, máximo, media e
    desvio padrão nas colunas do dataset.
    :param dataset: banco original de dados
    """
    c = []
    for j in range(len(dataset[0]) - 1):
        c.append([])
        for i in range(len(dataset)):
            c[j].append(float(dataset[i][j]))
    for i in range(len(c)):
        print(f'min    C{i}: %.3f' % min(c[i]))
        print(f'max    C{i}: %.3f' % max(c[i]))
        print(f'média  C{i}: %.3f' % st.mean(c[i]))
        print(f'desvio padrão C{i}: %.3f' % st.stdev(c[i]))
        print('-' * 50)

def ex_matriz(dataset):
    """
    Extrai a matriz do dataset, passando os valores
    numéricos para float, mantendo os valores string
    na sua forma original.
    :param dataset: banco original de dados
    :return: matriz formatada dos dados
    """
    m = []
    for i in range(len(dataset)):
        m.append([])
        for j in range(len(dataset[0])):
            if j != len(dataset[0]) - 1:
                m[i].append(float(dataset[i][j]))
            else: m[i].append(dataset[i][j])
    return m

def dist(a, b): return abs(a - b)

def cos_dist(v1, v2):
    """
    Calcula os cossenos vetoriais entre
	um vetor e cada registro de uma matriz.
    :param v1: um vetor
    :param v2: uma matriz
    :return: lista com cossenos vetoriais
    """
    dist = []
    for i in range(len(v2)):
        dist.append(abs(1 - 
			vet.cos_v(v1, v2[i][:4])))
    return dist

def euc_dist(v1, v2):
    """
    Calcula as distâncias euclidianas vetoriais
    entre um vetor e cada registro de uma matriz.
    :param v1: um vetor
    :param v2: uma matriz
    :return: lista com distâncias euclidianas vetoriais
    """
    dist = []
    for i in range(len(v2)):
        dist.append(vet.dist_euc_v(v1, v2[i][:4]))
    return dist

def knn(k, v1, v2, f):
    """
    Realiza o IBk k-NN entre uma linha do
    dataset e uma base particionada de dados.

    ** Descrição de regiões de código **

    (1): Preenche uma lista com as distâncias calculadas.
    (2): Cria pares com cada distância e sua índide na lista
    e por conseguinte, na matriz v2. Par: (distância, índice).
    (3): Ordena crescentemente os pares pelas distâncias.
    (4): Itera entre o intervalo [0, k[ e cria uma lista
    com os k-vizinhos mais próximos
    (5): O '1' (um) de z[i][1] indica o índice de cada par,
    assim o algoritmo salta no índice apontado em v2[índice]
    para extrair o argumento classificatório da flor
    por v2[índice][-1], para somar na lista dos vizinhos.
    :param k: valor de "k"
    :param v1: um vetor
    :param v2: uma matriz particionada de dados
    :return: uma lista com os 'k' vizinhos mais próximos
    """
    l = f(v1, v2) # (1)
    z = list(zip(l, range(len(l)))) # (2)
    z.sort(); r = [] # (3)
    for i in range(k): r.append(v2[z[i][1]][-1]) # (4), (5)
    return r

def moda_knn(l):
    """
    Obtém uma moda dado um vetor de vizinhos mais próximos.

    ** Descrição de regiões de código **

    (1): Define um vetor de 3 posições, começando em 0.
    (2): Itera na faixa dos vizinhos mais próximos
    e atribui a cada classe de flor detectada uma posição
    no vetor m, incrementando cada posição, em cada
    ocorrência das classes.
    (3): Pesquisa o índice do maior elemento de m e retorna
    a moda (classe do maior elemento), sendo que para
    "k" ímpar sempre haverá um maior e para "k" par, a moda
    será o primeiro maior elemento e atualiza o maior.
    :param l: lista de vizinhos mais próximos
    :return: a moda entre os vizinhos
    """
    m = [0, 0, 0] # (1)
    for i in range(len(l)): # (2)
        if l[i] == 'Iris-setosa': m[0] += 1
        elif l[i] == 'Iris-versicolor': m[1] += 1
        else: m[2] += 1
    max = m[0]; p = 0
    for i in range(len(m)):
        if (max < m[i]): max = m[i]; p = i # (3)
    if p == 0: return 'Iris-setosa'
    elif p == 1: return 'Iris-versicolor'
    else: return 'Iris-virginica'

def split_rdata(pc, v, f, k = 1):
    """
    Faz o "corte" do dataset dado um percentual
    de corte (pc), gerando uma base de testes
    para o algoritmo knn, que irá ser testado com
    o restante que sobrou do corte para a base

    ** Descrição de regiões de código **

    (1): Uso de variáveis globais "base" e "p",
    para posterior uso de "p" e opcional uso de "base"
    (2): Calcula o tamanho da base (p) dado um percentual
    (3): "Embaralha" os registros do dataset
    (4): Copia os registros da dataset para uma base
    no intervalo [0, p[, "separa" a base do testset
    (5): Itera na faixa do testset e roda o knn, e para cada knn é
    feita uma escolha da moda, gerando uma entrada
    o predito, que é a lista das modas do testset
    :param pc: a percentagem de corte
    :param v: a matriz de dados (dataset)
    :param k: valor de "k" para o knn (por padrão, k = 1)
    :return: o predito (lista de modas resultantes do teste)
    """
    global base, p; pred = [] # (1)
    p = int(round(pc * len(v))) # (2)
    random.shuffle(v); base = v[:p] # (3), (4) e ((5) - linha abaixo)
    for i in range(p, len(v)): pred.append(moda_knn(knn(k, v[i][:4], base, f)))
    return pred

def accuracy(testset, predic):
    """
    Determina o total e o percentual de acertos
    do knn com relação à base (acurácia).
    :param testset: o subconjunto registro da matriz
    de dados que ficou de "fora" da base
    :param predic: o predito
    :return: acurácia (precisão do algoritmo)
    """
    c = 0
    # Cada classe na mesma posição do predito é um acerto
    for i in range(len(testset)):
        if testset[i][-1] == predic[i]: c += 1
    # print('Acertos: %d de %d' %(c, len(testset)))
    return (c / float(len(testset))) * 100.0

def med_acc(nt, dataset, pc, f, k = 1):
    """
    Algoritmo de médias de execução. Executa
    várias vezes o IBK KNN e retira uma média
    da acurácia das execuções
    :param nt: número de vezes que é feita a execução
    do IBK
    :param dataset: a matriz de dados
    :param pc: percentual de corte
    :param k: valor "k" do knn
    :return: a média das acurácias
    """
    m = 0; global p
    for i in range(nt):
        pred = split_rdata(pc, dataset, f, k)
        m += accuracy(dataset[p:], pred)
    return m / float(nt)

if __name__ == '__main__':
    lines = abre_csv('iris.csv')
    matriz = ex_matriz(lines)
    pc = 0.66
    p = int(round(pc * len(matriz)))
    ne = 100; 
    f = cos_dist  # f = euc_dist
    print(f'Para {ne} execuções: ')
    print('Média das acurácias (k = 1): %.2f%%' % med_acc(ne, matriz, 0.66, f))
    print('Média das acurácias (k = 4): %.2f%%' % med_acc(ne, matriz, 0.66, f, 4))
    print('Média das acurácias (k = 8): %.2f%%' % med_acc(ne, matriz, 0.66, f, 8))

    """
    IBK KNN - Testes em 100 execuções
    Distância euclidiana:
        Média das acurácias (k = 1): 95.63%
        Média das acurácias (k = 4): 94.96%
        Média das acurácias (k = 8): 96.20%

    Distância por cosseno:
        Média das acurácias (k = 1): 96.14%
        Média das acurácias (k = 4): 96.51%
        Média das acurácias (k = 8): 96.63%
        
    *Comparação com o Weka (Acurácias):
        (k = 1): 96.07%
        (k = 4): 96.07%
        (k = 8): 98.03%
    """