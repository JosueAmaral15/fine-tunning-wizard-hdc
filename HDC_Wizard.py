'''
Author name: Josué Amaral
Description: Hyperdimensional Computing (HDC) Wizard Classifier
Date: 2025-07-14
Version: 1.0

This code implements a Hyperdimensional Computing (HDC) Wizard Classifier for
student use, which uses hyperdimensional vectors to classify data. It includes
methods for binding, bundling, and encoding data, as well as training and
predicting classes using HDC techniques. It also includes a controller class to
manage the data processing and evaluation of the model. This code is designed
to be modular and extensible, allowing for easy integration with various
datasets and classification tasks.

Hyperdimensional Computing (HDC) Wizard Classifier

This code is part of a larger project that explores the use of HDC for
classification tasks, including the implementation of a Wizard Dictionary
classifier. The code is structured to allow for easy modification and
extension, making it suitable for research and experimentation in the field of
HDC.

This code is released under the MIT License. You are free to use, modify, and
distribute this code as long as you include the original copyright notice and
this permission notice in any copies or substantial portions of the code. The
code is provided "as is", without warranty of any kind, express or implied,
including but not limited to the warranties of merchantability, fitness for a
particular purpose, and noninfringement. In no event shall the authors or
copyright holders be liable for any claim, damages, or other liability, whether
in an action of contract, tort, or otherwise, arising from, out of, or in
connection with the code or the use or other dealings in the code.

'''

from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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
    
    def train(self, X, y):
        prototipos = defaultdict(list)
        #print(f"y: {y}")
        for exemplo, classe in zip(X, y):
            #print(f"exemplo: {exemplo}, classe: {classe}")
            vetor = self._codificar_exemplo(exemplo)
            prototipos[classe].append(vetor)

        self.prototipos_classes = {
            classe: self.assemble.bundling(vetores) for classe, vetores in prototipos.items()
        }

    def predict(self, X):
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
    
    
    
    
# — encoder Termômetro —
class TermometerEncoder:
    def __init__(self, quantidade_bits):
        # Define a quantidade de bits de resolução desejada para o encoder
        self.quantidade_bits = quantidade_bits
        self.valor_minimo = None
        self.valor_maximo = None
        self.incremento = None

    def adjust(self, matriz_entrada):
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
    def __init__(self, total_de_bits, bits_per_address):
        # Garante que o total de bits seja múltiplo do tamanho do endereço
        assert total_de_bits % bits_per_address == 0
        self.bits_endereco = bits_per_address
        self.quantidade_unidades = total_de_bits // self.bits_endereco
        # Inicializa os filtros para cada unidade
        self.filtros = [FiltroDicionario() for _ in range(self.quantidade_unidades)]
    
    def train(self, vetor_binario):
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
    
    def __init__(self, total_entries, classes_number, bits_per_address, usar_branqueamento=False):
        self.usar_branqueamento = usar_branqueamento
        # Calcula o número de bits de preenchimento necessários
        self.number_extra_bits = ((total_entries // bits_per_address) * bits_per_address - total_entries) % bits_per_address
        self.total_entries = total_entries + self.number_extra_bits
        # Gera ordem aleatória dos bits (incluindo os extras)
        self.ordem_bits = np.arange(self.total_entries)
        #print(f"Total de bits: {self.total_entries}, Bits extras: {self.number_extra_bits}, Ordem dos bits: {self.ordem_bits}")
        np.random.shuffle(self.ordem_bits)
        # Inicializa os discriminators para cada classe
        self.discriminators = [
            Discriminador(self.total_entries, bits_per_address)
            for _ in range(classes_number)
        ]

    def _preparar(self, vetor):
        # Preenche o vetor e reordena os bits
        vetor_preenchido = np.pad(vetor, (0, self.number_extra_bits))[self.ordem_bits]
        return vetor_preenchido

    def train(self, matriz_binaria, vetor_classes):
        #print("TRAIN")
        # Treina os discriminators com os vetores
        #print(f"Treinando {len(self.discriminators)} discriminadores com {len(matriz_binaria)} vetores. Vetores por classe: {vetor_classes}.\nmatriz_binaria: {matriz_binaria}.")
        for vetor, classe in zip(matriz_binaria, vetor_classes):
            #print(f"Treinando vetor ({len(vetor)}): {vetor}, classe: {classe}.")
            self.discriminators[classe].train(self._preparar(vetor))

        if self.usar_branqueamento:
            # Calcula os escores médios para branqueamento
            escores = []
            for vetor, classe in zip(matriz_binaria, vetor_classes):
                escore = self.discriminators[classe].pontuar(self._preparar(vetor))
                escores.append(escore)
            self.limiar_branqueamento = ceil(np.mean(escores))
            # Atribui limiar aos discriminators
            for discriminator in self.discriminators:
                discriminator.bran = self.limiar_branqueamento

    def predict(self, matriz_binaria):
        #print("PREDICT")
        predictions = []
        for vetor in matriz_binaria:
            # Calcula escore de todos os discriminators
            escores = [discriminator.pontuar(self._preparar(vetor)) for discriminator in self.discriminators]
            # Aplica branqueamento se necessário
            if self.usar_branqueamento:
                escores = [escore if escore >= getattr(discriminator, 'bran', 0) else 0 for escore, discriminator in zip(escores, self.discriminators)]
            # Seleciona o índice da maior pontuação
            predictions.append(int(np.argmax(escores)))
        return np.array(predictions)

# Classe Controladora Principal
class ControllerMain:
    
    def process_input_data(self,dataset_identifier, normalize = False, first_normalize = True, termometer_encode = False, sample_limits=None):
        
        # Busca dataset pelo identificador da UCI
        dataset = fetch_ucirepo(id=dataset_identifier)
        
        # Seleciona somente colunas numéricas
        if dataset.data.targets is None:
            print(f"Dataset '{dataset_identifier}' não tem rótulo. Pulando.")
            return None, None, None, None

        matriz_X = dataset.data.features.select_dtypes(include=[np.number]).dropna(axis=1).values
        vetor_y = dataset.data.targets.iloc[:,0].astype('category').cat.codes.values
        #nomes_colunas = dataset.data.features.columns.tolist()
        classes_name = dataset.data.targets.iloc[:,0].astype('category').cat.categories.tolist()
        classes_number = len(np.unique(vetor_y))
                
        # Limitar a quantidade de amostras, se desejado
        if sample_limits is not None:
            matriz_X = matriz_X[:sample_limits]
            vetor_y = vetor_y[:sample_limits]
        
        if normalize:
            scaler = MinMaxScaler()
            matriz_X = scaler.fit_transform(matriz_X)
            
        # Divide entre treino e teste
        #print(f"vetor_y: {vetor_y}")
        X_train, X_test, y_train, y_test = train_test_split(matriz_X, vetor_y, test_size=0.3, random_state=0, stratify=vetor_y)
        #print(f"X_train: {X_train}, X_test: {X_test}, y_train: {y_train}, y_test: {y_test}")

        if termometer_encode:
            encoder = TermometerEncoder(20).adjust(X_train)
            
        if first_normalize:
            if normalize:
                # Normaliza os dados
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            if termometer_encode:
                # Codifica com termômetro
                X_train = encoder.binarizar(X_train)
                X_test = encoder.binarizar(X_test)
        else:
            if termometer_encode:
                # Codifica com termômetro
                X_train = encoder.binarizar(X_train)
                X_test = encoder.binarizar(X_test)
                
            if normalize:
                # Normaliza os dados
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
        
        if termometer_encode:
            input_total_count = X_train.shape[1]
        else:
            input_total_count = len(matriz_X)
        
        return X_train, X_test, y_train, y_test, classes_name, classes_number, Counter(y_train), Counter(y_test), input_total_count
    
    def avaliar_modelo(self, real_vector, vector_predicted, classes_name, titulo="Resultados da Classificação", show_confusion_matrix=True):
        
        # Imprime o relatório de classificação
        acuracia = accuracy_score(real_vector, vector_predicted)
        print("Relatório de Classificação:\nAcuracia: {:.2f}%".format(acuracia*100))
        print(classification_report(real_vector, vector_predicted, target_names=classes_name))
        
        if show_confusion_matrix:
            # Gera a matriz de confusão
            matriz_confusao = confusion_matrix(real_vector, vector_predicted)
            plt.figure(figsize=(10, 8))
            sns.heatmap(matriz_confusao, annot=True, fmt='g', cmap='Blues', xticklabels=classes_name, yticklabels=classes_name)
            plt.xlabel("Classe Predita")
            plt.ylabel("Classe verdadeira ou real")
            plt.title(f"Matriz de Confusão - {titulo}")
            plt.tight_layout()
            plt.show()
    
    def run_model(self, X_train, y_train, X_test, y_test, classes_name, model, model_name):

        print(f"Iniciando treinamento do modelo {model_name}...")
        model.train(X_train, y_train)
        print("Treinamento concluído.\nPrevendo...")
        predictions = model.predict(X_test)
        print("Previsão concluída.\nAvaliando modelo...")
        self.avaliar_modelo(y_test, predictions, classes_name, titulo=f"Resultados da Classificação com o modelo {model_name}", show_confusion_matrix=True)
        print(f"Avaliação do modelo {model_name} concluída.\n")
        
# Programa principal para executar o código
if __name__ == "__main__":
    PRINT_OTHER_OBSERVATIONS = False  # Variável para controlar a impressão de observações adicionais
    DIMENSION = 10000  # Definir a dimensionalidade dos vetores hiperdimensionais com tamanho típico em HDC:10000
    N_NIVEIS = 10  # Níveis de codificação termômetro
    HDC_MODELS_NAME = ["HDC - Record-based", "HDC - N-gram-based"]
    HDC_MODES = ['record', 'ngram']
    WIZARD_MESSAGE = ['Modelo wizard sem branqueamento', 'Modelo wizard com branqueamento']
    BLEACHING_MODE = [False, True]
    
    print_message = lambda X_train, X_test, classes_name, classes_number, quantity_y_train, quantity_y_test, input_total_count: f"""
Total de classes: {classes_number}, nomes das classes: {classes_name}
Quantidade de atributos/features: {X_train.shape[1]}, quantidade de amostras: {X_train.shape[0]}
Quantidade de amostras de treino: {len(X_train)}, quantidade de amostras de teste: {len(X_test)}
Total de amostras: {len(X_train) + len(X_test)}
total de colunas/features: {input_total_count}
Quantidade de amostras de treino por classe: {quantity_y_train}
Quantidade de amostras de teste por classe: {quantity_y_test}
        """
    
    print(f"Dimensão dos vetores: {DIMENSION}")
    
    datasets = {
        'iris': 53,
        #'adult': 2,
        #'secondary_mushroom' : 848,
        #'cdc_diabetes_health': 891,
    }
    
    controller = ControllerMain()
    
    for dataset, id_dataset in datasets.items():
        print(f"""
Nome da classe: {dataset}, id da classe: {id_dataset}
Dimensão dos vetores: {DIMENSION}
Quantidade de níveis de codificação: {N_NIVEIS}
        """)
        
        print("\n=== Classificação com Computação Hiperdimensional (HDC) ===")
        
        X_train, X_test, y_train, y_test, classes_name, classes_number, quantity_y_train, quantity_y_test, input_total_count = controller.process_input_data(id_dataset, normalize = True, first_normalize = True, termometer_encode = False)
        print_message(X_train, X_test, classes_name, classes_number, quantity_y_train, quantity_y_test, input_total_count)
        
        for model_name, mode in zip (HDC_MODELS_NAME, HDC_MODES):
            print(f"\n=== Classificação com {model_name} ===")
            hdc_record = HDCClassificador(d_dimensao=DIMENSION, n_niveis=N_NIVEIS, modo=mode)
            controller.run_model(X_train, y_train, X_test, y_test, classes_name, hdc_record, model_name)
            
        # Utilizando o modelo Wizard Dictionary para a classificação dos datasets
        print("\n=== Classificação com Wizard Dictionary ===")
        
        X_train, X_test, y_train, y_test, classes_name, classes_number, quantity_y_train, quantity_y_test, input_total_count = controller.process_input_data(id_dataset, normalize = False, first_normalize = False, termometer_encode = True)
        print_message(X_train, X_test, classes_name, classes_number, quantity_y_train, quantity_y_test, input_total_count)
        #print(f"Total de bits: {input_total_count}, Bits por endereço: 8, Total de classes: {classes_number}")
        for model_name, com_branqueamento in zip(WIZARD_MESSAGE,BLEACHING_MODE):
            print(f"\n=== Classificação com {model_name} ===")
            wisard = ClassificadorWisard(input_total_count, classes_number, bits_per_address = 8, usar_branqueamento=com_branqueamento)
            controller.run_model(X_train, y_train, X_test, y_test, classes_name, wisard, model_name)
        
