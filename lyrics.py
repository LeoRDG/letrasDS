import re
import nltk
import string
import unicodedata
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.util import ngrams
from wordcloud import WordCloud
from time import perf_counter as pf
from dataclasses import dataclass, field
from lexicalrichness import LexicalRichness
from collections import Counter, defaultdict

nltk.download("punkt")
nltk.download("reuters")

contador_global: int = 0
pontuação: str = string.punctuation
lexico_re: re = re.compile(r'^(.+)\..+N0=([\d-]+)')
stopwords: set = set(i.strip() for i in open("stopwords").readlines())
lexico: dict = {lexico_re.match(linha).groups() for linha in open("SentiLex-PT02/SentiLex-lem-PT02.txt").readlines() if lexico_re.match(linha)}

def remover_acentos(text: str) -> str:
    """ Remove o acento de letras """
    text = text.replace("\n", " ")
    return ''.join( c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn" )

def calcular_sentimento(palavras :list) -> float:
    """ Calcula o sentimento de uma lista de palavras """
    polaridades = [lexico.get(p, 0) for p in palavras if p in lexico]
    return round(sum(polaridades) / len(polaridades), 3) if polaridades else 0.0

def calcular_robustez(letra: str, media_tamanho_palavra: float = None) -> float:
    """Calcula a robustez de um texto"""
    lex = LexicalRichness(letra)
    
    try:
        ttr = lex.ttr
        mtld = lex.mtld(threshold=.72)
        mattr = lex.mattr(window_size=25)
    except ZeroDivisionError:
        return 0.0
    except ValueError:
        mattr=lex.mattr(window_size=1)
    
    # MTLD ranges 10-200+, TTR is 0-1
    # Combine them using simple z-scores or min-max scaling
    score = (
        ttr * 1.5 +        # TTR: high = more diverse
        (mtld / 100) +     # scale down for comparability
        (mattr * 1.5)      # MATTR: high = better local richness
    )

    # Add mean word length if available
    if media_tamanho_palavra is not None:
        score += media_tamanho_palavra * 0.5

    return round(score, 3)


@dataclass
class DadosClasseBase:
    dados: pd.DataFrame = field(repr=False)
    size: int = field(repr=True, init=False)
    nome: str = field(init=False, default="")

    def sentimento(self):
        return self.dados["sentimento"].median()

    def n_gram(self, n: int = 2, top: int = 10) -> dict[tuple[str]: dict[str: int]]:
        """Calcula frases mais ocorrentes"""
        print(f"ngram {self.nome or "ALL"} {n}")

        total_counter = Counter()
        song_tracker = defaultdict(set)

        for i, row in self.dados.iterrows():
            letra = row["letra"]

            if not isinstance(letra, str):
                continue
            
            tokens = nltk.word_tokenize(letra)
            song_ngrams = set()

            for ng in ngrams(tokens, n):
                if any(len(word) < 4 for word in ng):
                    continue

                total_counter[ng] += 1
                song_ngrams.add(ng)

            for ng in song_ngrams:
                song_tracker[ng].add(i)

        result = {
            ng: {"count": total_counter[ng], "songs": len(song_tracker[ng])} 
            for ng in total_counter
        }

        result = sorted(result.items(), key= lambda x: x[1]["count"], reverse=True)
        return result[:top]       

    def __post_init__(self):
        self.size = len(self.dados)
    
    def palavras_mais_comuns(self, num: int=0) -> Counter:
        """Retorna uma lista com tuples com as palavras mais comuns entre musicas desse genero"""
        total_palavras = []
        
        for palavras in self.dados["palavras"]:
            if isinstance(palavras, list):
                total_palavras.extend(set(palavras))

        return Counter(total_palavras) if num <= 0 else Counter(total_palavras).most_common(num)

    def palavras_mais_ocorrentes(self, num: int=0) -> Counter:
        """Retorna um tuple com as palavras mais ocorrentes entre musicas de um genero"""
        total_palavras = []
        
        for palavras in self.dados["palavras"]:
            if isinstance(palavras, list):
                total_palavras.extend(palavras)

        return Counter(total_palavras) if num <= 0 else Counter(total_palavras).most_common(num)

    def get_generos(self) -> Counter:
        """ Retorna um Counter com os gêneros e a quantidade de músicas desse gênero"""
        data = [item for item in self.dados["genero"]]
        return Counter(data).most_common()
    
    def get_artistas(self) -> Counter:
        data = [a for a in self.dados["artista"]]
        return Counter(data).most_common()

    def numero_medio_palavras(self) -> int:
        """Retorna o numero medio de palavras na letra"""
        count = 0
        count2 = 0
        for texto in self.dados["letra"]:
            if isinstance(texto, str):
                count += len(texto.split())
                count2 += 1
        return count / count2

    def TTR(self):
        pass


@dataclass
class Artista(DadosClasseBase):
    nome: str = field(repr=True)

@dataclass
class Genero(DadosClasseBase):
    nome: str = field(repr=True)

@dataclass
class Letras(DadosClasseBase):
    def __post_init__(self):
        self.processar_dados()

    ()
    def processar_dados(self) -> None:
        """Processa self.dados para serem trabalhados.
        É chamada quando essa classe é criada"""
        
        # Renomear colunas
        self.dados.rename( inplace=True, columns={
                "Nome da Música": "musica",
                "Artista": "artista",
                "Gênero Musical": "genero",
                "Letra da Música": "letra"
             }) 
        
        self.dados = self.dados.dropna() # Remove entradas vazias
        self.processar_generos()
        self.dados["artista"] = self.dados["artista"].apply(lambda a: str(a).strip().lower()) # Normaliza os nomes dos artistas
        self.processar_letras()

        # self.dados["palavras"] = self.dados["letra"].apply(lambda letra: [i for i in re.findall( r"[a-zãõẽĩũáéúíóçêôêîâü]{3,}", str(letra).lower()) if i and i not in stopwords])

    ()
    def processar_generos(self) -> None:
        """ Processa os generos """
        def processar(cell: str) -> str:
            """ Normaliza os nomes na coluna gênero, remove acentos e - """
            cell = remover_acentos(str(cell)).strip().replace("-", " ").lower()
            return "gauchas" if "gauchas" in cell else cell

        self.dados["genero"] = self.dados["genero"].apply(processar)
        self.dados = self.dados.groupby("genero").filter(lambda g: len(g) >= 80) # Remove generos com menos de 80 músicas
        self.dados = self.dados.loc[self.dados["genero"].apply(lambda g: not("/" in g or ";" in g))] # Remove generos compostos
        self.dados = self.dados.copy()

    def processar_letras(self) -> None:
        """ Processa as letras das músicas """

        global contador_global

        def processar(row):
            global contador_global
            if contador_global % 1000 == 0:
                print(f"Calculando médias {contador_global}/{len(self.dados)}", end="\r")
            contador_global += 1
            letra:str = row["letra"]

            # Pula letras vazias (não deve mais acontecer)
            if pd.isnull(letra) or not isinstance(letra, str):
                return pd.Series({
                'palavras': None,
                'total_palavras': None,
                'palavras_unicas': None,
                'tamanho_medio': None,
                'sentimento': None,
                "robustez": None
            }) 

            # separa as palavras ema lista
            palavras = re.findall(r"[a-zãõẽĩũáéúíóçêôêîâü]+", letra.lower())
            
            # Extração de alguns insights simples
            total_palavras = len(palavras)
            palavras_unicas = len(set(palavras))
            tamanho_medio = round(sum(len(p) for p in palavras) / total_palavras, 3)
            palavras_relevantes = [p for p in palavras if len(p) >= 3 and p not in stopwords]
            sentimento = calcular_sentimento(palavras_relevantes)
            robustez = calcular_robustez(letra)

            
            # Cria novas colunas com os insights
            return pd.Series({
                'palavras': palavras_relevantes,
                'total_palavras': int(total_palavras),
                'palavras_unicas': int(palavras_unicas),
                'tamanho_medio': float(tamanho_medio),
                'sentimento': float(sentimento),
                "robustez": float(robustez)
            })


        def remover_duplicatas():
            # remove letras duplicadas
            print("- removendo duplicatas", end='\r')
            self.dados["sem_acentos"] = self.dados["letra"].dropna().apply(remover_acentos)
            self.dados = self.dados.drop_duplicates("sem_acentos")
            self.dados.drop(columns = ["sem_acentos"], inplace=True)
            
        contador_global = 0
        remover_duplicatas()
        

        # Limpa os dados das letras e retorna novas colunas
        self.dados [[
            "palavras",
            'total_palavras',
            "palavras_unicas",
            "tamanho_medio",
            "sentimento",
            "robustez"
            ]] = self.dados.apply(processar, axis=1)
        print("DONE")

    def genero(self, genero:str) -> Genero:
        """Retorna a classe Genero com todas as entradas desse genero"""
        data = self.dados[self.dados["genero"].apply(lambda g: genero in g)]
        return Genero(dados=data, nome=genero)

    def artista(self, nome) -> Artista:
        """Retorna a classe Artista com todas as entradas que contém esse artista"""
        data = self.dados[self.dados["artista"] == nome]
        return Artista(dados=data, nome=nome)
    

l = Letras(pd.read_csv("letras.csv"))
generos_desejados = ["sertanejo","funk carioca", "infantil", "heavy metal", "regional"]

print("_"*60)
print("Gêneros")
for gen, n in l.get_generos():
    print(f"{gen:>15} {n}")
print("_"*60)

print()
print("_"*60)
print("Artistas")
for a, n in l.get_artistas():
    print(f"{gen:>15} {n}")
print("_"*60)

print()
for gen in generos_desejados:
    print(gen)
    dados_genero = l.genero(gen)
    
    print('.'*5)
    print("mais comuns")
    for i in dados_genero.palavras_mais_comuns(10):
        print(i)


    print('.'*5)
    print("mais ocorrentes")
    for i in dados_genero.palavras_mais_ocorrentes(10):
        print(i)
    
    print("."*5)
    print("frases mais comuns/ocorrentes")
    for i, j in zip(dados_genero.n_gram(2), dados_genero.n_gram(3)):
        print(f"{i} {" ":<20} {j}")
    print()
    print("_"*60)
        



###################################################### 
##################### GRÁFICOS #######################
######################################################  

# plt.figure(figsize=(10, 6))
# plt.xlim(0, 800)
# sns.histplot(l.dados['total_palavras'].dropna(), bins=50, kde=True)
# plt.title("Distribuição do número de palavras por música")
# plt.xlabel("Total de palavras")
# plt.ylabel("Número de músicas")
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.scatterplot(
#     data=l.dados,
#     x='total_palavras',
#     y='palavras_unicas',
#     alpha=0.6
# )
# plt.title("Relação entre total de palavras e palavras únicas")
# plt.xlabel("Total de palavras")
# plt.ylabel("Palavras únicas")
# plt.grid(True)
# plt.show()


# l.dados = l.dados[l.dados['genero'].str.lower().isin(generos)]

# plt.figure(figsize=(12, 6))
# sns.boxplot(data=l.dados, x='genero', y='total_palavras')
# plt.xticks(rotation=45)
# plt.title("Distribuição do total de palavras por gênero")
# plt.xlabel("Gênero")
# plt.ylabel("Total de palavras")
# plt.tight_layout()
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(12, 6))
# sns.boxplot(data=l.dados, x='genero', y='robustez')
# plt.xticks(rotation=45)
# plt.title("Robustez lexical por gênero")
# plt.xlabel("Gênero")
# plt.ylabel("Pontuação de robustez")
# plt.tight_layout()
# plt.grid(True)
# plt.show()


# Normaliza o nome do gênero para evitar problemas de capitalização
# df_filtrado = l.dados[l.dados['genero'].str.lower().isin(generos_desejados)]
# generos = df_filtrado['genero'].unique()
# sns.heatmap(df_filtrado[['total_palavras', 'palavras_unicas', 'tamanho_medio', 'sentimento', 'robustez']].corr(), annot=True, cmap='coolwarm')
# plt.title("Correlação entre métricas")
# plt.show()

# for i in l.n_gram(2, 5):
#     print(i)
# for i in l.n_gram(3, 5):
#     print(i)
# for i in l.n_gram(4, 5):
#     print(i)
# for i in l.n_gram(5, 5):
#     print(i)

# for genero in generos_desejados:
#     dados = l.genero(genero)

#     print(dados, dados.sentimento())
#     count = dados.palavras_mais_ocorrentes(5)
#     songs = dados.palavras_mais_comuns(5)

#     for i, j in zip(count, songs):
#         print(i, j)

#     for i in dados.n_gram(2, 5):
#         print(i)
#     for i in dados.n_gram(3, 5):
#         print(i)
#     print("_____________________________________________")
#     input()

# contagem_generos = l.dados['genero'].value_counts().sort_values(ascending=False)
# plt.figure(figsize=(12, 6))
# sns.barplot(
#     x=contagem_generos.values[:20],
#     y=contagem_generos.index[:20],
#     palette="mako"
# )
# plt.title("Top 20 gêneros com mais músicas no dataset")
# plt.xlabel("Número de músicas")
# plt.ylabel("Gênero")
# plt.tight_layout()
# plt.show()
