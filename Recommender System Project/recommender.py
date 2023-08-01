# Standard library imports
import os
import sys
import time
import warnings

# Third-party imports
import numpy as np
import pandas as pd
import pickle
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk import word_tokenize
from nltk.corpus import stopwords
from pympler import asizeof
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM, TextDatasetForNextSentencePrediction, Trainer, TrainingArguments

warnings.filterwarnings("ignore")

# TF-IDF Vectorizer Content Recommender class
class ContentRecommenderTFIDF:

    def __init__(self, movies_df, k=100):
        self.movies_df = movies_df
        self.tfidf = self.train_tfidf()
        self.top_k_similar_movies = self.get_top_k_similar_movies(k)
        self.scaler = MinMaxScaler()

    def train_tfidf(self):
        tfidf_vector = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_df=1305, min_df=5, sublinear_tf=True)
        tfidf_matrix = tfidf_vector.fit_transform(self.movies_df['combined_text'])
        return csr_matrix(tfidf_matrix)

    def get_top_k_similar_movies(self, k):
        similarity_matrix = cosine_similarity(self.tfidf)
        top_k_similar_movies = {}

        for i in range(similarity_matrix.shape[0]):
            top_k_indices = np.argsort(similarity_matrix[i])[::-1][1:k+1]
            top_k_similar_movies[i] = top_k_indices

        return top_k_similar_movies

    def recommend(self, movie_id, top_n=10):

        movie_index = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
        top_k_indices = self.top_k_similar_movies[movie_index]
        
        recommendations = self.movies_df.iloc[top_k_indices][['movieId', 'title', 'vote_count', 'vote_average', 'score', 'sentiment']]
        movie_similarities = cosine_similarity(self.tfidf[movie_index], self.tfidf[top_k_indices]).flatten()

        recommendations['cosine_similarity'] = movie_similarities

        recommendations['vote_count'] = recommendations['vote_count'].astype('int')
        recommendations['vote_average'] = recommendations['vote_average'].astype('int')

        input_movie_sentiment = self.movies_df.loc[movie_index, 'sentiment']
        recommendations['sentiment_difference'] = np.abs(recommendations['sentiment'] - input_movie_sentiment)

        C = recommendations['vote_average'].mean()
        m = recommendations['vote_count'].quantile(0.6)

        qualified = recommendations[(recommendations['vote_count'] >= m) & (recommendations['vote_count'].notnull()) & (recommendations['vote_average'].notnull())]

        qualified.loc[:, ['score', 'sentiment_difference', 'cosine_similarity']] = self.scaler.fit_transform(qualified[['score', 'sentiment_difference', 'cosine_similarity']])
        qualified.loc[:, 'combined_score'] = qualified['score'] * 0.1 + qualified['cosine_similarity'] * 0.7 + (1 - qualified['sentiment_difference']) * 0.2
        qualified = qualified.sort_values('combined_score', ascending=False).head(top_n)

        return qualified


# TF-IDF Vectorizer Optimized Content Recommender class
class ContentRecommenderTFIDFOptimized:
    def __init__(self, movies_df, k=100):
        self.movies_df = movies_df
        self.tfidf = self.train_tfidf()
        self.top_k_similar_movies = self.get_top_k_similar_movies(k)
        self.scaler = MinMaxScaler()

    def train_tfidf(self):
        tfidf_vector = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_df=1305, min_df=5, sublinear_tf=True)
        tfidf_matrix = tfidf_vector.fit_transform(self.movies_df['combined_text'])
        return csr_matrix(tfidf_matrix)

    def get_top_k_similar_movies(self, k):
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree', metric='euclidean').fit(self.tfidf)
        return nbrs

    def recommend(self, movie_id, top_n=10):
        movie_index = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
        distances, top_k_indices = self.top_k_similar_movies.kneighbors(self.tfidf[movie_index])
        top_k_indices = top_k_indices[0][1:]

        recommendations = self.movies_df.iloc[top_k_indices][['title', 'vote_count', 'vote_average', 'score', 'sentiment']]

        recommendations['cosine_similarity'] = 1 - distances[0][1:]**2

        recommendations['vote_count'] = recommendations['vote_count'].astype('int')
        recommendations['vote_average'] = recommendations['vote_average'].astype('int')

        input_movie_sentiment = self.movies_df.loc[movie_index, 'sentiment']
        recommendations['sentiment_difference'] = np.abs(recommendations['sentiment'] - input_movie_sentiment)

        C = recommendations['vote_average'].mean()
        m = recommendations['vote_count'].quantile(0.6)

        qualified = recommendations[(recommendations['vote_count'] >= m) & (recommendations['vote_count'].notnull()) & (recommendations['vote_average'].notnull())]

        qualified.loc[:, ['score', 'sentiment_difference', 'cosine_similarity']] = self.scaler.fit_transform(qualified[['score', 'sentiment_difference', 'cosine_similarity']])
        qualified.loc[:, 'combined_score'] = qualified['score'] * 0.1 + qualified['cosine_similarity'] * 0.7 + (1 - qualified['sentiment_difference']) * 0.2
        qualified = qualified.sort_values('combined_score', ascending=False).head(top_n)

        return qualified


# Count Vector Content Recommender class
class ContentRecommenderCountVec:
    def __init__(self, movies_df, k=100):
        self.movies_df = movies_df
        self.count_vec = self.train_count_vec()
        self.top_k_similar_movies = self.get_top_k_similar_movies(k)
        self.scaler = MinMaxScaler()

    def train_count_vec(self):
        count_vector = CountVectorizer(stop_words='english', ngram_range=(1, 3), max_df=1305, min_df=5)
        count_matrix = count_vector.fit_transform(self.movies_df['combined_text'])
        return csr_matrix(count_matrix)

    def get_top_k_similar_movies(self, k):
        similarity_matrix = cosine_similarity(self.count_vec)
        top_k_similar_movies = {}

        for i in range(similarity_matrix.shape[0]):
            top_k_indices = np.argsort(similarity_matrix[i])[::-1][1:k+1]
            top_k_similar_movies[i] = top_k_indices

        return top_k_similar_movies

    def recommend(self, movie_id, top_n=10):
        movie_index = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
        top_k_indices = self.top_k_similar_movies[movie_index]

        recommendations = self.movies_df.iloc[top_k_indices][['title', 'vote_count', 'vote_average', 'score', 'sentiment']]
        movie_similarities = cosine_similarity(self.count_vec[movie_index], self.count_vec[top_k_indices]).flatten()

        recommendations['cosine_similarity'] = movie_similarities

        recommendations['vote_count'] = recommendations['vote_count'].astype('int')
        recommendations['vote_average'] = recommendations['vote_average'].astype('int')

        input_movie_sentiment = self.movies_df.loc[movie_index, 'sentiment']
        recommendations['sentiment_difference'] = np.abs(recommendations['sentiment'] - input_movie_sentiment)

        C = recommendations['vote_average'].mean()
        m = recommendations['vote_count'].quantile(0.6)

        qualified = recommendations[(recommendations['vote_count'] >= m) & (recommendations['vote_count'].notnull()) & (recommendations['vote_average'].notnull())]

        qualified.loc[:, ['score', 'sentiment_difference', 'cosine_similarity']] = self.scaler.fit_transform(qualified[['score', 'sentiment_difference', 'cosine_similarity']])
        qualified.loc[:, 'combined_score'] = qualified['score'] * 0.1 + qualified['cosine_similarity'] * 0.7 + (1 - qualified['sentiment_difference']) * 0.2
        qualified = qualified.sort_values('combined_score', ascending=False).head(top_n)

        return qualified


# Count Vector Optimized Content Recommender class
class ContentRecommenderCountVecOptimized:
    def __init__(self, movies_df, k=100):
        self.movies_df = movies_df
        self.count_vector = self.train_count_vector()
        self.top_k_similar_movies = self.get_top_k_similar_movies(k)
        self.scaler = MinMaxScaler()

    def train_count_vector(self):
        count_vector = CountVectorizer(stop_words='english', ngram_range=(1, 3), max_df=1305, min_df=5)
        count_matrix = count_vector.fit_transform(self.movies_df['combined_text'])
        return csr_matrix(count_matrix)

    def get_top_k_similar_movies(self, k):
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree', metric='euclidean').fit(self.count_vector)
        return nbrs

    def recommend(self, movie_id, top_n=10):
        movie_index = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
        distances, top_k_indices = self.top_k_similar_movies.kneighbors(self.count_vector[movie_index])
        top_k_indices = top_k_indices[0][1:]

        recommendations = self.movies_df.iloc[top_k_indices][['title', 'vote_count', 'vote_average', 'score', 'sentiment']]

        recommendations['cosine_similarity'] = 1 - distances[0][1:]**2

        recommendations['vote_count'] = recommendations['vote_count'].astype('int')
        recommendations['vote_average'] = recommendations['vote_average'].astype('int')

        input_movie_sentiment = self.movies_df.loc[movie_index, 'sentiment']
        recommendations['sentiment_difference'] = np.abs(recommendations['sentiment'] - input_movie_sentiment)

        C = recommendations['vote_average'].mean()
        m = recommendations['vote_count'].quantile(0.6)

        qualified = recommendations[(recommendations['vote_count'] >= m) & (recommendations['vote_count'].notnull()) & (recommendations['vote_average'].notnull())]

        qualified.loc[:, ['score', 'sentiment_difference', 'cosine_similarity']] = self.scaler.fit_transform(qualified[['score', 'sentiment_difference', 'cosine_similarity']])
        qualified.loc[:, 'combined_score'] = qualified['score'] * 0.1 + qualified['cosine_similarity'] * 0.7 + (1 - qualified['sentiment_difference']) * 0.2
        qualified = qualified.sort_values('combined_score', ascending=False).head(top_n)

        return qualified


# Word2Vec Content Recommender class
class ContentRecommenderW2V:
    def __init__(self, movies_df, k=100):
        self.movies_df = movies_df
        self.preprocessed_text = self.movies_df['combined_text'].apply(self.preprocess_text).tolist()
        self.word2vec_model = self.train_word2vec_model()
        self.movies_embeddings = self.get_movie_embeddings()
        self.top_k_similar_movies = self.get_top_k_similar_movies(k)
        self.scaler = MinMaxScaler()

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        return [word for word in tokens if word.isalpha() and word not in stop_words]

    def train_word2vec_model(self):
        model = Word2Vec(self.preprocessed_text, vector_size=300, window=5, min_count=3, workers=4, sg=1, epochs=10)
        return model

    def get_movie_embeddings(self):
        movie_embeddings = []
        for text in self.preprocessed_text:
            embeddings = np.mean([self.word2vec_model.wv[word] for word in text if word in self.word2vec_model.wv], axis=0)
            movie_embeddings.append(embeddings)
        return np.vstack(movie_embeddings)

    def get_top_k_similar_movies(self, k):
        similarity_matrix = cosine_similarity(self.movies_embeddings)
        top_k_similar_movies = {}

        for i in range(similarity_matrix.shape[0]):
            top_k_indices = np.argsort(similarity_matrix[i])[::-1][1:k+1]
            top_k_similar_movies[i] = top_k_indices

        return top_k_similar_movies

    def recommend(self, movie_id, top_n=10):
        movie_index = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
        top_k_indices = self.top_k_similar_movies[movie_index]

        recommendations = self.movies_df.iloc[top_k_indices][['title', 'vote_count', 'vote_average', 'score', 'sentiment']]
        movie_similarities = [cosine_similarity(self.movies_embeddings[movie_index].reshape(1, -1), self.movies_embeddings[idx].reshape(1, -1)).flatten()[0] for idx in top_k_indices]

        recommendations['cosine_similarity'] = movie_similarities

        recommendations['vote_count'] = recommendations['vote_count'].astype('int')
        recommendations['vote_average'] = recommendations['vote_average'].astype('int')

        input_movie_sentiment = self.movies_df.loc[movie_index, 'sentiment']
        recommendations['sentiment_difference'] = np.abs(recommendations['sentiment'] - input_movie_sentiment)

        C = recommendations['vote_average'].mean()
        m = recommendations['vote_count'].quantile(0.6)

        qualified = recommendations[(recommendations['vote_count'] >= m) & (recommendations['vote_count'].notnull()) & (recommendations['vote_average'].notnull())]

        qualified.loc[:, ['score', 'sentiment_difference', 'cosine_similarity']] = self.scaler.fit_transform(qualified[['score', 'sentiment_difference', 'cosine_similarity']])
        qualified.loc[:, 'combined_score'] = qualified['score'] * 0.1 + qualified['cosine_similarity'] * 0.7 + (1 - qualified['sentiment_difference']) * 0.2
        qualified = qualified.sort_values('combined_score', ascending=False).head(top_n)

        return qualified


# Word2Vec Optimized Content Recommender class
class ContentRecommenderW2VOptimized:
    def __init__(self, movies_df, k=100):
        self.movies_df = movies_df
        self.wv_model = self.train_word2vec()
        self.top_k_similar_movies = self.get_top_k_similar_movies(k)
        self.scaler = MinMaxScaler()

    def tokenize(self, text):
        tokens = word_tokenize(text)
        tokens = [t.lower() for t in tokens if t.isalpha()]
        return tokens

    def train_word2vec(self):
        stop_words = set(stopwords.words("english"))
        sentences = self.movies_df['combined_text'].apply(lambda x: [word for word in self.tokenize(x) if word not in stop_words])
        wv_model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)
        return wv_model

    def get_top_k_similar_movies(self, k):
        movie_embeddings = np.array([self.get_movie_embedding(movie) for movie in self.movies_df['combined_text']])
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='brute', metric='cosine').fit(movie_embeddings)
        return nbrs

    def get_movie_embedding(self, text):
        words = self.tokenize(text)
        words = [word for word in words if word in self.wv_model.wv]
        if len(words) == 0:
            return np.zeros(self.wv_model.vector_size)
        return np.mean(self.wv_model.wv[words], axis=0)

    def recommend(self, movie_id, top_n=10):
        movie_index = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
        movie_embedding = self.get_movie_embedding(self.movies_df.loc[movie_index, 'combined_text'])
        distances, top_k_indices = self.top_k_similar_movies.kneighbors([movie_embedding])
        top_k_indices = top_k_indices[0][1:]

        recommendations = self.movies_df.iloc[top_k_indices][['title', 'vote_count', 'vote_average', 'score', 'sentiment']]

        recommendations['cosine_similarity'] = 1 - distances[0][1:]

        recommendations['vote_count'] = recommendations['vote_count'].astype('int')
        recommendations['vote_average'] = recommendations['vote_average'].astype('int')

        input_movie_sentiment = self.movies_df.loc[movie_index, 'sentiment']
        recommendations['sentiment_difference'] = np.abs(recommendations['sentiment'] - input_movie_sentiment)

        C = recommendations['vote_average'].mean()
        m = recommendations['vote_count'].quantile(0.6)

        qualified = recommendations[(recommendations['vote_count'] >= m) & (recommendations['vote_count'].notnull()) & (recommendations['vote_average'].notnull())]
        qualified.loc[:, ['score', 'sentiment_difference', 'cosine_similarity']] = self.scaler.fit_transform(qualified[['score', 'sentiment_difference', 'cosine_similarity']])
        qualified.loc[:, 'combined_score'] = qualified['score'] * 0.1 + qualified['cosine_similarity'] * 0.7 + (1 - qualified['sentiment_difference']) * 0.2
        qualified = qualified.sort_values('combined_score', ascending=False).head(top_n)
        
        return qualified


# Doc2Vec Content Recommender class
class ContentRecommenderD2V:
    def __init__(self, movies_df, k=100):
        self.movies_df = movies_df
        self.preprocessed_text = self.movies_df['combined_text'].apply(self.preprocess_text).tolist()
        self.doc2vec_model = self.train_doc2vec_model()
        self.movies_embeddings = self.get_movie_embeddings()
        self.top_k_similar_movies = self.get_top_k_similar_movies(k)
        self.scaler = MinMaxScaler()

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        return [word for word in tokens if word.isalpha() and word not in stop_words]

    def train_doc2vec_model(self):
        tagged_documents = [TaggedDocument(words=text, tags=[str(i)]) for i, text in enumerate(self.preprocessed_text)]
        model = Doc2Vec(tagged_documents, vector_size=300, window=5, min_count=3, workers=4, epochs=10)
        return model

    def get_movie_embeddings(self):
        movie_embeddings = [self.doc2vec_model.dv[str(i)] for i in range(len(self.preprocessed_text))]
        return np.vstack(movie_embeddings)

    def get_top_k_similar_movies(self, k):
        similarity_matrix = cosine_similarity(self.movies_embeddings)
        top_k_similar_movies = {}

        for i in range(similarity_matrix.shape[0]):
            top_k_indices = np.argsort(similarity_matrix[i])[::-1][1:k+1]
            top_k_similar_movies[i] = top_k_indices

        return top_k_similar_movies

    def recommend(self, movie_id, top_n=10):
        movie_index = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
        top_k_indices = self.top_k_similar_movies[movie_index]

        recommendations = self.movies_df.iloc[top_k_indices][['title', 'vote_count', 'vote_average', 'score', 'sentiment']]
        movie_similarities = [cosine_similarity(self.movies_embeddings[movie_index].reshape(1, -1), self.movies_embeddings[idx].reshape(1, -1)).flatten()[0] for idx in top_k_indices]

        recommendations['cosine_similarity'] = movie_similarities

        recommendations['vote_count'] = recommendations['vote_count'].astype('int')
        recommendations['vote_average'] = recommendations['vote_average'].astype('int')

        input_movie_sentiment = self.movies_df.loc[movie_index, 'sentiment']
        recommendations['sentiment_difference'] = np.abs(recommendations['sentiment'] - input_movie_sentiment)

        C = recommendations['vote_average'].mean()
        m = recommendations['vote_count'].quantile(0.6)

        qualified = recommendations[(recommendations['vote_count'] >= m) & (recommendations['vote_count'].notnull()) & (recommendations['vote_average'].notnull())]

        qualified.loc[:, ['score', 'sentiment_difference', 'cosine_similarity']] = self.scaler.fit_transform(qualified[['score', 'sentiment_difference', 'cosine_similarity']])
        qualified.loc[:, 'combined_score'] = qualified['score'] * 0.1 + qualified['cosine_similarity'] * 0.7 + (1 - qualified['sentiment_difference']) * 0.2
        qualified = qualified.sort_values('combined_score', ascending=False).head(top_n)

        return qualified
    
# Doc2Vec Optimized Content Recommender class
class ContentRecommenderD2VOptimized:
    def __init__(self, movies_df, k=100):
        self.movies_df = movies_df
        self.dv_model = self.train_doc2vec()
        self.top_k_similar_movies = self.get_top_k_similar_movies(k)
        self.scaler = MinMaxScaler()

    def tokenize(self, text):
        tokens = word_tokenize(text)
        tokens = [t.lower() for t in tokens if t.isalpha()]
        return tokens

    def train_doc2vec(self):
        stop_words = set(stopwords.words("english"))
        tagged_documents = [
            TaggedDocument(
                words=[word for word in self.tokenize(text) if word not in stop_words],
                tags=[str(index)]
            )
            for index, text in self.movies_df['combined_text'].iteritems()
        ]
        dv_model = Doc2Vec(tagged_documents, vector_size=100, window=5, min_count=5, workers=4)
        return dv_model

    def get_top_k_similar_movies(self, k):
        movie_embeddings = np.array([self.get_movie_embedding(index) for index in self.movies_df.index])
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='brute', metric='cosine').fit(movie_embeddings)
        return nbrs

    def get_movie_embedding(self, index):
        return self.dv_model.dv[str(index)]

    def recommend(self, movie_id, top_n=10):
        movie_index = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
        movie_embedding = self.get_movie_embedding(movie_index)
        distances, top_k_indices = self.top_k_similar_movies.kneighbors([movie_embedding])
        top_k_indices = top_k_indices[0][1:]

        recommendations = self.movies_df.iloc[top_k_indices][['title', 'vote_count', 'vote_average', 'score', 'sentiment']]

        recommendations['cosine_similarity'] = 1 - distances[0][1:]

        recommendations['vote_count'] = recommendations['vote_count'].astype('int')
        recommendations['vote_average'] = recommendations['vote_average'].astype('int')

        input_movie_sentiment = self.movies_df.loc[movie_index, 'sentiment']
        recommendations['sentiment_difference'] = np.abs(recommendations['sentiment'] - input_movie_sentiment)

        C = recommendations['vote_average'].mean()
        m = recommendations['vote_count'].quantile(0.6)

        qualified = recommendations[(recommendations['vote_count'] >= m) & (recommendations['vote_count'].notnull()) & (recommendations['vote_average'].notnull())]
        qualified.loc[:, ['score', 'sentiment_difference', 'cosine_similarity']] = self.scaler.fit_transform(qualified[['score', 'sentiment_difference', 'cosine_similarity']])
        qualified.loc[:, 'combined_score'] = qualified['score'] * 0.1 + qualified['cosine_similarity'] * 0.7 + (1 - qualified['sentiment_difference']) * 0.2
        qualified = qualified.sort_values('combined_score', ascending=False).head(top_n)
        
        return qualified