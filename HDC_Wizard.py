from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from ucimlrepo import fetch_ucirepo
from collections import Counter
from math import ceil

class HDCAssemble:
    
    # Função para gerar um vetor binário aleatório {-1, 1}
    @staticmethod
    def gerar_vetor_binario(dimension = 10000):
        return np.random.choice([-1, 1], size=dimension)

    #Binding: vinculativo;
    #Binding → Liga partes específicas para formar estruturas complexas (Ex: cor + forma → maçã vermelha)
    # Função de binding: combinação (associação) via multiplicação elemento a elemento (equivalente a XOR binário)
    @staticmethod
    def binding(v1, v2):
        return v1 * v2

    #Bundling: agrupamento.
    #Bundling → Resume várias instâncias ou ocorrências para formar um "prototipo" ou representação média (Ex: ideia geral do que é uma maçã).
    # Função de bundling: combinação por soma e threshold.
    # Bundling (combinação) por soma + threshold
    @staticmethod
    def bundling(vetores):
        soma = np.sum(vetores, axis=0)
        return np.where(soma >= 0, 1, -1)

    # Similaridade cosseno
    @staticmethod
    def similaridade(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # Função de codificação termômetro para variáveis contínuas
    @staticmethod
    def codificacao_termometro(valor, n_niveis):
        limite_superior = n_niveis - 1
        nivel_ativo = int(round(valor * limite_superior))
        vetor = np.full(n_niveis, -1)
        vetor[:nivel_ativo + 1] = 1
        return vetor

# Classe do classificador HDC
class HDCClassificador:
    def __init__(self, d_dimensao=10000, n_niveis=10, modo = 'record'):
        self.DIMENSION = d_dimensao
        self.n_niveis = n_niveis
        self.modo = modo  # 'record' ou 'ngram'
        self.vetores_atributos = {}  # Vetores aleatórios para cada atributo e nível
        self.vetores_posicoes = {}   # Vetores aleatórios para posições dos atributos
        self.prototipos_classes = {}  # Vetores protótipos de cada classe
        self.assemble = HDCAssemble()  # Instância da classe de operações HDC
        
    def _codificar_exemplo(self, exemplo):
        vetores = []

        if self.modo == 'record':
            for i, valor in enumerate(exemplo):
                if i not in self.vetores_posicoes:
                    self.vetores_posicoes[i] = self.assemble.gerar_vetor_binario(self.DIMENSION)
                vetor_posicao = self.vetores_posicoes[i]

                for nivel in range(self.n_niveis):
                    chave = (i, nivel)
                    if chave not in self.vetores_atributos:
                        self.vetores_atributos[chave] = self.assemble.gerar_vetor_binario(self.DIMENSION)

                vetor_termo = self.assemble.codificacao_termometro(valor, self.n_niveis)
                vetores_nivel = []
                for nivel in range(self.n_niveis):
                    if vetor_termo[nivel] == 1:
                        vetor_nivel = self.vetores_atributos[(i, nivel)]
                        vetor_bind = self.assemble.binding(vetor_nivel, vetor_posicao)
                        vetores_nivel.append(vetor_bind)

                if vetores_nivel:
                    vetor_atributo = self.assemble.bundling(vetores_nivel)
                    vetores.append(vetor_atributo)

        elif self.modo == 'ngram':
            for i in range(len(exemplo) - 1):
                valor1, valor2 = exemplo[i], exemplo[i + 1]

                if i not in self.vetores_posicoes:
                    self.vetores_posicoes[i] = self.assemble.gerar_vetor_binario(self.DIMENSION)
                if (i + 1) not in self.vetores_posicoes:
                    self.vetores_posicoes[i + 1] = self.assemble.gerar_vetor_binario(self.DIMENSION)

                vetor_pos1 = self.vetores_posicoes[i]
                vetor_pos2 = self.vetores_posicoes[i + 1]

                for nivel1 in range(self.n_niveis):
                    chave1 = (i, nivel1)
                    if chave1 not in self.vetores_atributos:
                        self.vetores_atributos[chave1] = self.assemble.gerar_vetor_binario(self.DIMENSION)
                for nivel2 in range(self.n_niveis):
                    chave2 = (i + 1, nivel2)
                    if chave2 not in self.vetores_atributos:
                        self.vetores_atributos[chave2] = self.assemble.gerar_vetor_binario(self.DIMENSION)

                termo1 = self.assemble.codificacao_termometro(valor1, self.n_niveis)
                termo2 = self.assemble.codificacao_termometro(valor2, self.n_niveis)

                vetores_ngram = []
                for n1 in range(self.n_niveis):
                    for n2 in range(self.n_niveis):
                        if termo1[n1] == 1 and termo2[n2] == 1:
                            bind1 = self.assemble.binding(self.vetores_atributos[(i, n1)], vetor_pos1)
                            bind2 = self.assemble.binding(self.vetores_atributos[(i + 1, n2)], vetor_pos2)
                            ngram_bind = self.assemble.binding(bind1, bind2)
                            vetores_ngram.append(ngram_bind)

                if vetores_ngram:
                    vetor_ngram_final = self.assemble.bundling(vetores_ngram)
                    vetores.append(vetor_ngram_final)

        return self.assemble.bundling(vetores)
    
    def treinar(self, X, y):
        prototipos = defaultdict(list)
        for exemplo, classe in zip(X, y):
            vetor = self._codificar_exemplo(exemplo)
            prototipos[classe].append(vetor)

        self.prototipos_classes = {
            classe: self.assemble.bundling(vetores) for classe, vetores in prototipos.items()
        }

    def prever(self, X):
        predicoes = []
        for exemplo in X:
            vetor = self._codificar_exemplo(exemplo)
            melhor_classe = None
            maior_sim = -np.inf
            for classe, prototipo in self.prototipos_classes.items():
                sim = self.assemble.similaridade(vetor, prototipo)
                if sim > maior_sim:
                    maior_sim = sim
                    melhor_classe = classe
            predicoes.append(melhor_classe)
        return predicoes
    
# Função de avaliação completa
def avaliar_modelo(nome, y_true, y_pred, nomes_classes, show_confusion_matrix=True):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"\n=== {nome} ===")
    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")

    if show_confusion_matrix:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=nomes_classes, yticklabels=nomes_classes)
        plt.title(f'Matriz de Confusão - {nome}')
        plt.ylabel('Verdadeiro')
        plt.xlabel('Previsto')
        plt.show()

def processar_id(identificador_dataset, limite_amostras=None):
    # Busca dataset pelo identificador da UCI
    dataset = fetch_ucirepo(id=identificador_dataset)

    if dataset.data.targets is None:
        print(f"Dataset '{identificador_dataset}' não tem rótulo. Pulando.")
        return None, None, None, None

    matriz_X = dataset.data.features.select_dtypes(include=[np.number]).dropna(axis=1).values
    vetor_y = dataset.data.targets.iloc[:, 0].astype('category').cat.codes.values
    nomes_classes = dataset.data.targets.iloc[:, 0].astype('category').cat.categories.tolist()
    numero_classes = len(np.unique(vetor_y))

    # Limitar a quantidade de amostras, se desejado
    if limite_amostras is not None:
        matriz_X = matriz_X[:limite_amostras]
        vetor_y = vetor_y[:limite_amostras]

    # Divide entre treino e teste
    scaler = MinMaxScaler()
    X_normalizado = scaler.fit_transform(matriz_X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalizado, vetor_y, test_size=0.3, random_state=42, stratify=vetor_y
    )

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    total_entradas = X_train.shape[1]

    return nomes_classes, numero_classes, X_train, y_train, Counter(y_train), X_test, y_test, Counter(y_test), total_entradas

    
# — Codificador Termômetro —
class CodificadorTermometro:
    def __init__(self, quantidade_bits):
        # Define a quantidade de bits de resolução desejada para o codificador
        self.quantidade_bits = quantidade_bits
        self.valor_minimo = None
        self.valor_maximo = None
        self.incremento = None

    def ajustar(self, matriz_entrada):
        # Armazena o valor mínimo de cada coluna
        self.valor_minimo = matriz_entrada.min(axis=0)
        # Armazena o valor máximo de cada coluna
        self.valor_maximo = matriz_entrada.max(axis=0)
        # Calcula o passo de incremento por nível
        self.incremento = (self.valor_maximo - self.valor_minimo) / (self.quantidade_bits - 1 + 1e-9)
        return self

    def binarizar(self, matriz_entrada):
        numero_amostras, numero_variaveis = matriz_entrada.shape
        quantidade_bits = self.quantidade_bits
        # Inicializa a matriz binarizada
        matriz_saida = np.zeros((numero_amostras, numero_variaveis * quantidade_bits), dtype=bool)
        # Percorre cada elemento da matriz original
        for indice_amostra in range(numero_amostras):
            for indice_variavel in range(numero_variaveis):
                # Normaliza e converte o valor da variável
                valor_normalizado = (matriz_entrada[indice_amostra, indice_variavel] - self.valor_minimo[indice_variavel]) / (self.incremento[indice_variavel] + 1e-9)
                nivel_ativado = int(min(max(round(valor_normalizado), 0), quantidade_bits))
                # Liga os bits até o nível ativado
                matriz_saida[indice_amostra, indice_variavel * quantidade_bits : indice_variavel * quantidade_bits + nivel_ativado] = True
        return matriz_saida

# — Filter Bloom leve (Dictionary WiSARD) —
class FiltroDicionario:
    def __init__(self):
        # Inicializa um conjunto vazio para armazenar os padrões
        self.memoria = set()

    def adicionar(self, tupla):
        # Adiciona uma nova tupla à memória
        self.memoria.add(tuple(tupla))

    def pertence(self, tupla):
        # Verifica se a tupla está presente na memória
        return tuple(tupla) in self.memoria

# — Discriminador para cada classe —
class Discriminador:
    def __init__(self, total_de_bits, bits_por_endereco):
        # Garante que o total de bits seja múltiplo do tamanho do endereço
        assert total_de_bits % bits_por_endereco == 0
        self.bits_endereco = bits_por_endereco
        self.quantidade_unidades = total_de_bits // self.bits_endereco
        # Inicializa os filtros para cada unidade
        self.filtros = [FiltroDicionario() for _ in range(self.quantidade_unidades)]
    
    def treinar(self, vetor_binario):
        total_bits_esperado = self.quantidade_unidades * self.bits_endereco
        tamanho_atual = vetor_binario.size

        if tamanho_atual < total_bits_esperado:
            faltando = total_bits_esperado - tamanho_atual
            vetor_binario = np.concatenate([vetor_binario, np.zeros(faltando, dtype=int)])
        elif tamanho_atual > total_bits_esperado:
            vetor_binario = vetor_binario[:total_bits_esperado]

        partes = vetor_binario.reshape(self.quantidade_unidades, self.bits_endereco)
        for indice, parte in enumerate(partes):
            self.filtros[indice].adicionar(parte)

    def pontuar(self, vetor_binario):
        # Divide o vetor e soma os acertos por unidade
        partes = vetor_binario.reshape(self.quantidade_unidades, self.bits_endereco)
        return sum(int(self.filtros[i].pertence(partes[i])) for i in range(self.quantidade_unidades))

# — Classificador Dictionary WiSARD —
class ClassificadorWisard:
    
    def __init__(self, total_de_entradas, numero_classes, bits_por_endereco, usar_branqueamento=False):
        self.usar_branqueamento = usar_branqueamento
        # Calcula o número de bits de preenchimento necessários
        self.numero_bits_extras = ((total_de_entradas // bits_por_endereco) * bits_por_endereco - total_de_entradas) % bits_por_endereco
        self.total_de_entradas = total_de_entradas + self.numero_bits_extras
        # Gera ordem aleatória dos bits (incluindo os extras)
        self.ordem_bits = np.arange(self.total_de_entradas)
        np.random.shuffle(self.ordem_bits)
        # Inicializa os discriminadores para cada classe
        self.discriminadores = [
            Discriminador(self.total_de_entradas, bits_por_endereco)
            for _ in range(numero_classes)
        ]

    def _preparar(self, vetor):
        # Preenche o vetor e reordena os bits
        vetor_preenchido = np.pad(vetor, (0, self.numero_bits_extras))[self.ordem_bits]
        return vetor_preenchido

    def treinar(self, matriz_binaria, vetor_classes):
        # Treina os discriminadores com os vetores
        for vetor, classe in zip(matriz_binaria, vetor_classes):
            self.discriminadores[classe].treinar(self._preparar(vetor))

        if self.usar_branqueamento:
            # Calcula os escores médios para branqueamento
            escores = []
            for vetor, classe in zip(matriz_binaria, vetor_classes):
                escore = self.discriminadores[classe].pontuar(self._preparar(vetor))
                escores.append(escore)
            self.limiar_branqueamento = ceil(np.mean(escores))
            # Atribui limiar aos discriminadores
            for discriminador in self.discriminadores:
                discriminador.bran = self.limiar_branqueamento

    def prever(self, matriz_binaria):
        previsoes = []
        for vetor in matriz_binaria:
            # Calcula escore de todos os discriminadores
            escores = [discriminador.pontuar(self._preparar(vetor)) for discriminador in self.discriminadores]
            # Aplica branqueamento se necessário
            if self.usar_branqueamento:
                escores = [escore if escore >= getattr(discriminador, 'bran', 0) else 0 for escore, discriminador in zip(escores, self.discriminadores)]
            # Seleciona o índice da maior pontuação
            previsoes.append(int(np.argmax(escores)))
        return np.array(previsoes)

# Classe Controladora Principal
class ControllerMain:
    
    def processar_id(self,identificador_dataset):
        # Busca dataset pelo identificador da UCI
        dataset = fetch_ucirepo(id=identificador_dataset)
        # Seleciona somente colunas numéricas
        if dataset.data.targets is None:
            print(f"Dataset '{identificador_dataset}' não tem rótulo. Pulando.")
            return None, None, None, None

        matriz_X = dataset.data.features.select_dtypes(include=[np.number]).dropna(axis=1).values
        vetor_y = dataset.data.targets.iloc[:,0].astype('category').cat.codes.values
        #nomes_colunas = dataset.data.features.columns.tolist()
        nomes_classes = dataset.data.targets.iloc[:,0].astype('category').cat.categories.tolist()
        numero_classes = len(np.unique(vetor_y))

        # Divide entre treino e teste
        X_treino, X_teste, y_treino, y_teste = train_test_split(matriz_X, vetor_y, test_size=0.3, random_state=0, stratify=vetor_y)

        # Codifica com termômetro
        codificador = CodificadorTermometro(20).ajustar(X_treino)
        X_treino_bin = codificador.binarizar(X_treino)
        X_teste_bin = codificador.binarizar(X_teste)

        total_entradas = X_treino_bin.shape[1]

        return nomes_classes, numero_classes, total_entradas, X_treino, X_treino_bin, X_teste, X_teste_bin, y_treino, y_teste

    def avaliar_modelo(self, vetor_real = None, vetor_predito = None, nomes_das_classes = None, titulo="Resultados da Classificação", show_confusion_matrix=True):
        
        if vetor_real is None:
            vetor_real = self.y_teste
        
        if vetor_predito is None:
            vetor_predito = self.previsoes
        
        if nomes_das_classes is None:
            nomes_das_classes = self.nomes_classes

        if show_confusion_matrix:
            # Gera a matriz de confusão
            matriz_confusao = confusion_matrix(vetor_real, vetor_predito)
            plt.figure(figsize=(10, 8))
            sns.heatmap(matriz_confusao, annot=True, fmt='g', cmap='Blues', xticklabels=nomes_das_classes, yticklabels=nomes_das_classes)
            plt.xlabel("Classe Predita")
            plt.ylabel("Classe verdadeira ou real")
            plt.title(f"Matriz de Confusão - {titulo}")
            plt.tight_layout()
            plt.show()
        
        # Imprime o relatório de classificação
        acuracia = accuracy_score(vetor_real, vetor_predito)
        print("Relatório de Classificação:\nAcuracia: {:.2f}%".format(acuracia*100))
        print(classification_report(vetor_real, vetor_predito, target_names=nomes_das_classes))
    
    def runWizard(self, X_treino_bin, y_treino, X_teste_bin, com_branqueamento = False):
        wisard = ClassificadorWisard(total_entradas, numero_classes, bits_por_endereco=8, usar_branqueamento=com_branqueamento)
        wisard.treinar(X_treino_bin, y_treino)
        previsoes = wisard.prever(X_teste_bin)
        
        return previsoes
    
# Programa principal para executar o código
if __name__ == "__main__":
    PRINT_OTHER_OBSERVATIONS = False  # Variável para controlar a impressão de observações adicionais
    DIMENSION = 10000  # Definir a dimensionalidade dos vetores hiperdimensionais com tamanho típico em HDC:10000
    N_NIVEIS = 10  # Níveis de codificação termômetro
        
    print(f"Dimensão dos vetores: {DIMENSION}")
    
    datasets = {
        'iris': 53,
        #'adult': 2,
        #'secondary_mushroom' : 848,
        #'cdc_diabetes_health': 891,
    }
    
    for dataset, id_dataset in datasets.items():
        nomes_classes, numero_classes, X_train, y_train, quantity_y_train, X_test, y_test, quantity_y_test, total_entradas = processar_id(id_dataset)
        
        print(f"""
Nome da classe: {dataset}, total de colunas/features: {total_entradas}
Classes: {nomes_classes}, quantidade de classes: {numero_classes}
Quantidade de treino: {len(y_train)}, quantidade de teste: {len(y_test)}
Níveis de codificação: {N_NIVEIS}""")
    
        # Record-based
        hdc_record = HDCClassificador(d_dimensao=DIMENSION, n_niveis=N_NIVEIS, modo='record')
        print("Iniciando treinamento record...")
        hdc_record.treinar(X_train, y_train)
        print("Treinamento concluído.\nPrevendo...")
        pred_record = hdc_record.prever(X_test)
        print("Previsão concluída.\nAvaliando modelo...")
        avaliar_modelo("HDC - Record-based", y_test, pred_record, nomes_classes)
        print("Avaliação record concluída.\n")
        
        # N-gram based
        hdc_ngram = HDCClassificador(d_dimensao=DIMENSION, n_niveis=N_NIVEIS, modo='ngram')
        print("Iniciando treinamento N-gram...")
        hdc_ngram.treinar(X_train, y_train)
        print("Treinamento concluído.\nPrevendo...")
        pred_ngram = hdc_ngram.prever(X_test)
        print("Previsão concluída.\nAvaliando modelo...")
        avaliar_modelo("HDC - N-gram based", y_test, pred_ngram, nomes_classes)
        print("Avaliação N-gram concluída.\n")
        
    
    # Utilizando o modelo Wizard Dictionary para a classificação dos datasets
    print("\n=== Classificação com Wizard Dictionary ===")
    controller = ControllerMain()

    for nome_dataset, identificador in datasets.items():
        print(f"\nProcessando {nome_dataset} (id={identificador}) …")
        acuracia0, acuracia1, limiar, frequencia_classes = controller.processar_id(identificador)
        
        if acuracia0 is not None and acuracia1 is not None and limiar is not None and frequencia_classes is not None:
            print(f"Frequência das classes no treino: {frequencia_classes}")
            print(f"Acurácia sem branqueamento: {acuracia0:.4f}")
            print(f"Acurácia com branqueamento: {acuracia1:.4f} (limiar={limiar})")

        controller.avaliar_modelo(titulo = f"resultado de classificação de {nome_dataset}")
