import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re
import nltk
from typing import List, Tuple, Optional, Dict, Any
import logging

class TextPreprocessor:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self._setup_logging()
        self._download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)
            
    def _download_nltk_data(self):
        """Download required NLTK data if not present"""
        for package in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                nltk.download(package, quiet=True)
                
    def preprocess_text(self, text: str) -> str:
        """Preprocess text with optional debugging"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = text.split()
            
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        processed = ' '.join(tokens)
        if self.debug:
            self.logger.debug(f"Preprocessed text: {processed[:100]}...")
        
        return processed

class DocumentSimilarity:
    def __init__(self, max_features: int = 5000, min_df: int = 1, 
                 max_df: float = 1.0, debug: bool = False):
        """
        Initialize document similarity processor
        
        Args:
            max_features: Maximum number of features for TF-IDF
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            debug: Enable debug logging
        """
        self.debug = debug
        self._setup_logging()
        
        self.preprocessor = TextPreprocessor(debug=debug)
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\b\w+\b',
            ngram_range=(1, 1)
        )
        
    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)
            
    def fit_transform_documents(self, df: pd.DataFrame, 
                              text_column: str) -> pd.DataFrame:
        """Process documents and create TF-IDF matrix"""
        self.logger.debug("Processing documents...")
        
        # Preprocess texts
        processed_texts = df[text_column].apply(self.preprocessor.preprocess_text)
        if self.debug:
            self.logger.debug("\nSample processed texts:")
            for text in processed_texts.head(3):
                self.logger.debug(f"{text[:100]}...")
        
        # Fit and transform
        tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        feature_names = self.vectorizer.get_feature_names_out()
        
        if self.debug:
            self.logger.debug(f"\nVocabulary size: {len(feature_names)}")
            self.logger.debug(f"Sample terms: {feature_names[:10]}")
        
        # Create DataFrame with TF-IDF features
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=df.index,
            columns=[f'tfidf_{feat}' for feat in feature_names]
        )
        
        return pd.concat([df, tfidf_df], axis=1)
    
    def transform_query(self, query: str) -> np.ndarray:
        """Transform query text to TF-IDF vector"""
        processed_query = self.preprocessor.preprocess_text(query)
        if self.debug:
            self.logger.debug(f"\nProcessed query: '{processed_query}'")
        
        query_vector = self.vectorizer.transform([processed_query])
        query_array = query_vector.toarray()[0]
        
        if self.debug:
            self._log_query_stats(query_array)
            
        return query_array
    
    def _log_query_stats(self, query_array: np.ndarray):
        """Log query vector statistics if debug is enabled"""
        vocabulary = self.vectorizer.get_feature_names_out()
        non_zero_indices = np.nonzero(query_array)[0]
        
        self.logger.debug(f"\nQuery stats:")
        self.logger.debug(f"Vector norm: {np.linalg.norm(query_array):.6f}")
        self.logger.debug(f"Non-zero terms: {len(non_zero_indices)}")
        
        if len(non_zero_indices) > 0:
            self.logger.debug("Matching terms:")
            for idx in non_zero_indices:
                self.logger.debug(f"  {vocabulary[idx]}: {query_array[idx]:.6f}")

    def filter_document_vectors(self, document_vectors: np.ndarray, df_indices: List[Any], 
                            valid_indices: set) -> Tuple[np.ndarray, List[Any]]:
        """Filter document vectors to only include those with indices in valid_indices set"""
        # Create mask for valid indices
        mask = [idx in valid_indices for idx in df_indices]
        
        # Filter vectors and indices
        filtered_vectors = document_vectors[mask]
        filtered_indices = [idx for idx, keep in zip(df_indices, mask) if keep]
        
        return filtered_vectors, filtered_indices

    def find_similar_documents(self, query_vector: np.ndarray, 
                             document_vectors: np.ndarray,
                             df_indices: List[Any],
                             top_k: int = 5) -> Tuple[List[Any], List[float]]:
        """
        Find most similar documents using cosine similarity
        
        Returns:
            Tuple of (document indices, similarity scores)
            Empty lists if no similar documents found
        """
        if len(df_indices) != document_vectors.shape[0]:
            raise ValueError("Number of indices must match number of document vectors")
            
        # Compute similarities
        query_norm = np.linalg.norm(query_vector)
        doc_norms = np.linalg.norm(document_vectors, axis=1)
        
        # Handle zero vectors
        if query_norm == 0:
            self.logger.warning("Query vector is zero - no matches possible")
            return [], []
            
        # Compute cosine similarities efficiently
        similarities = np.zeros(len(document_vectors))
        non_zero_docs = doc_norms > 0
        
        if not any(non_zero_docs):
            self.logger.warning("No valid document vectors found")
            return [], []
            
        # Vectorized similarity computation
        similarities[non_zero_docs] = (
            document_vectors[non_zero_docs] @ query_vector / 
            (doc_norms[non_zero_docs] * query_norm)
        )
        
        # Get top k results
        valid_similarities = similarities[~np.isnan(similarities)]
        if len(valid_similarities) == 0:
            return [], []
            
        top_k = min(top_k, len(valid_similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [df_indices[i] for i in top_indices], similarities[top_indices].tolist()
    
    def save_model(self, filepath: str):
        """Save model state"""
        model_data = {
            'vectorizer': self.vectorizer,
            'preprocessor': self.preprocessor
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @staticmethod
    def load_model(filepath: str, debug: bool = False) -> 'DocumentSimilarity':
        """Load saved model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        processor = DocumentSimilarity(
            max_features=model_data['vectorizer'].max_features,
            debug=debug
        )
        processor.vectorizer = model_data['vectorizer']
        processor.preprocessor = model_data['preprocessor']
        return processor