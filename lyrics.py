import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
import string
import re
from dataclasses import dataclass, field
import unicodedata


punctuation = string.punctuation

def load_stopwords():
    stopwords = set()
    with open("stopwords") as f:
        for l in f.readlines():
            stopwords.add(l.strip())
    return stopwords

stopwords = load_stopwords()


def keep_row_genero(cell) -> bool:
    """Manter apenas as entradas com uma só categoria, exceto se for musicas gaúchas"""
    if ";" in cell:
        return any(("gaúcha" in i for i in cell.split(";")))
    if "/" in cell or cell == "nan":
        return False
    return True


def process_generos(cell):
    cell = remove_accents(str(cell)).strip().replace("-", " ")
    return "gauchas" if "gauchas" in cell else cell

def remove_accents(text: str) -> str:
    text = text.lower().replace("\n", " ")
    return ''.join( c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn" )


@dataclass
class DadosClasseBase:
    dados: pd.DataFrame = field(repr=False)
    size: int = field(repr=True, init=False)
    nome: str = field(init=False)

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
        
        valores = []
        data = self.dados["palavras"]
        data = self.dados['letra']

        for i in data:
            if isinstance(i, str):
                i = i.split()
            if not i:
                continue
            print(i)        
            palavras_unicas = set(i)

            valores.append(len(palavras_unicas) / len(i))
            

        return sum(valores) / len(valores)

@dataclass
class Artista(DadosClasseBase):
    nome: str

    
@dataclass
class Genero(DadosClasseBase):
    nome: str = field(repr=True)
    size: int = field(repr=True, init=False)

    def __post_init__(self):
        self.size = len(self.dados)


@dataclass
class Letras(DadosClasseBase):
    def __post_init__(self):
        self.processar_dados()

    def processar_dados(self) -> None:
        """Processa os dados para serem mais fácilmente trabalhados"""

        self.dados.rename( inplace=True, columns={
                "Nome da Música": "musica",
                "Artista": "artista",
                "Gênero Musical": "genero",
                "Letra da Música": "letra"
             })
        
        # Limpeza dos dados baseando em genero
        self.dados["genero"] = self.dados["genero"].apply(process_generos)
        self.dados = self.dados.loc[self.dados["genero"].apply(keep_row_genero)]
        self.dados = self.dados.copy()

        self.dados["artista"] = self.dados["artista"].apply(lambda item: str(item).strip().lower())

        # Remove letras duplicadas
        

        
        self.processar_letras()

        # self.dados["palavras"] = self.dados["letra"].apply(lambda letra: [i for i in re.findall( r"[a-zãõẽĩũáéúíóçêôêîâü]{3,}", str(letra).lower()) if i and i not in stopwords])

    def processar_letras(self) -> None:
        print("processando letras")
        t1 = time()
        global counter
        counter = 0
        def processar(row):
            global counter
            if counter % 20000 == 0:
                print(f"Calculando médias {counter}/{len(self.dados)}")
            counter += 1
            letra:str = row["letra"]

            # Pula letras vazias
            if pd.isnull(letra) or not isinstance(letra, str):
                return pd.Series({
                'total_palavras': None,
                'total_palavras_unicas': None,
                'media_tamanho_palavras': None,
                'palavras': None,
            }) 

            # separa as palavras ema lista
            palavras = re.findall(r"[a-zãõẽĩũáéúíóçêôêîâü]+", letra.lower())
            
            # Extração de alguns insights simples
            total_palavras = len(palavras)
            total_palavras_unicas = len(set(palavras))
            media_tamanho_palavras = sum(len(p) for p in palavras) / total_palavras
            palavras_relevantes = (p for p in palavras if len(p) >= 3 and p not in stopwords)

            # Cria novas colunas com os insights
            return pd.Series({
                'total_palavras': total_palavras,
                'total_palavras_unicas': total_palavras_unicas,
                'media_tamanho_palavras': media_tamanho_palavras,
                'palavras': palavras_relevantes,
            })

        # remove letras duplicadas
        print("- removendo duplicatas")
        self.dados["sem_acentos"] = self.dados["letra"].dropna().apply(remove_accents)
        self.dados = self.dados.drop_duplicates("sem_acentos")
        self.dados.drop(columns = ["sem_acentos"])

        # Limpa os dados das letras e retorna novas colunas
        self.dados [['total_palavras', "total_palavras_unicas", "media_tamanho_palavras", "palavras"]] = self.dados.apply(processar, axis=1)
        print(f"{len(self.dados)} letras processadas em {time()-t1:.1f} s")

    def genero(self, genero:str) -> Genero:
        """Retorna a classe Genero com todas as entradas desse genero"""
        data = self.dados[self.dados["genero"].apply(lambda g: genero in g)]
        return Genero(dados=data, nome=genero)

    def artista(self, nome) -> Artista:
        """Retorna a classe Artista com todas as entradas que contém esse artista"""
        data = self.dados[self.dados["artista"] == nome]
        return Artista(dados=data, nome=nome)
    

l = Letras(pd.read_csv("letras.csv"))
print(len(l.dados))
