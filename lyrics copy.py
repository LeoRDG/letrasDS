import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
import string

def preprocess(text):
    text = text.lower()

    for p in punctuation:
        text = text.replace(p, "")
    words = text.split()

    return [word for word in words if word not in stopwords_pt and word.isalpha()]

stopwords_pt = set()#set(stopwords.words("portuguese"))
with open("stopwords") as f:
    for l in f.readlines():
        stopwords_pt.add(l.strip())


punctuation = set(string.punctuation)
 
data = pd.read_csv("letras_genius.csv")
print(list(data))

all_genres = {}

def normalize_entry(entry):
    if isinstance(entry, float):
        return ""
    


    # cleanup the strings
    cleaned = [part.strip().replace("-", " ").lower() for part in entry.split(";")]

    for part in entry.split(";"):
        all_genres[part.strip().replace('-', ' ').lower()] = all_genres.get(part.strip().replace('-', ' ').lower(), 0) + 1
    
    

    # remove duplicates and sort
    return "; ".join(sorted(set(cleaned)))
        

# Normalize genres names
data['tag'] = data['tag'].apply(normalize_entry)


for n, i in enumerate(sorted(all_genres, key=lambda i: all_genres[i])):
    genre, count = (i, all_genres[i])
    print(f'{count:03d} - {genre}')
# for genre, count in data['Gênero Musical'].value_counts().items():

def most_common_words(genre, amount):
    genre_lyrics = data[data["tag"].str.contains(genre, case=False, na=False)]["lyrics"]
    
    all_words = []
    for lyrics in genre_lyrics.dropna():
        all_words.extend(preprocess(lyrics))

    word_freq = Counter(all_words)

    return word_freq.most_common(amount)


for i in most_common_words("pop", 20):
    print(i)

arti = Counter(data['tag'])
arti = arti.most_common()
for i in arti:
    print(i)



