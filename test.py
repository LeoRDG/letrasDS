import pandas as pd

input = "song_lyrics.csv"
out = "letras_genius.csv"

remove = ["features", "id", "language_cld3", "language_ft", "language"]

chunk = 100000

write_header = True

for chunk in pd.read_csv(input, chunksize=chunk):
    filtered = chunk[chunk['language'] == 'pt']
    filtered = filtered.drop(columns=remove)
    filtered.to_csv(out, mode='a', index=False, header=write_header)
    write_header = False