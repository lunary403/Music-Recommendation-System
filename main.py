import pandas as pd
df = pd.read_csv("spotify_millsongdata.csv")
df.head(5)

df.shape
df.isnull().sum()

df = df.sample(5000).drop('link', axis=1).reset_index(drop= True)

df['text'][0]
df['text'] = df['text'].str.lower().replace(r'^\w\s', ' ').replace(r'\n', ' ', regex = True)
df['text']
df

import nltk
nltk.download('punkt', quiet=True, raise_on_error=True)
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def token(txt):
    tokens = nltk.word_tokenize(txt)
    stemmer = PorterStemmer()
    stemmed_text = [stemmer.stem(word) for word in tokens]
    return " ".join(stemmed_text)

df['text'].apply(lambda x: token(x))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse import vstack

#df = pd.DataFrame(data)

# Sample a subset of the data
sample_size = 1000
df_sample = df.sample(sample_size, random_state=42)

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')

# Fit and transform the sampled data
matrix = tfidf_vectorizer.fit_transform(df_sample['text'])

# Compute cosine similarity
similarity = cosine_similarity(matrix)

# Print the similarity matrix
print("Cosine Similarity Matrix:\n", similarity)


similarity[0]

df[df['song']=='Ballad For A Friend'].index[0]

def recommender(song_name):
    idx = df[df['song']==song_name].index[0]
    distance = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x:x[1])
    song = []
    for song_index in distance[1:5]:
        song.append(df.iloc[song_index[0]].song)
    return song        

recommender("Ballad For A Friend")


import pickle

pickle.dump(similarity, open("similarity", "wb"))
pickle.dump(df, open("df", "wb"))
