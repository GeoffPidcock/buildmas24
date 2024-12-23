import pandas as pd
from pathlib import Path
import pickle
from typing import List, Tuple, Any
from src.similarity import DocumentSimilarity

class DocumentSimilarityManager:
    """Manager class to handle document processing and model persistence"""
    
    def __init__(self, 
                 model_dir: str = "data/models",
                 processed_dir: str = "data/processed",
                 debug: bool = False):
        self.model_dir = Path(model_dir)
        self.processed_dir = Path(processed_dir)
        self.debug = debug
        
        # Create directories if they don't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.similarity_engine = None
        self.document_vectors = None
        self.document_indices = None
        
    def train_model(self, 
                   df: pd.DataFrame,
                   text_column: str,
                   max_features: int = 5000,
                   min_df: int = 2,
                   max_df: float = 0.95) -> None:
        """Train new similarity model and process documents"""
        
        # Initialize and train similarity engine
        self.similarity_engine = DocumentSimilarity(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            debug=self.debug
        )
        
        # Process documents
        processed_df = self.similarity_engine.fit_transform_documents(df, text_column)
        
        # Store document vectors and indices
        tfidf_cols = [col for col in processed_df.columns if col.startswith('tfidf_')]
        self.document_vectors = processed_df[tfidf_cols].values
        self.document_indices = processed_df.index.tolist()
        
        # Save processed data
        self._save_processed_data(processed_df)
        self._save_model()
        
    def _save_model(self) -> None:
        """Save trained model and vectorizer"""
        model_path = self.model_dir / "similarity_model.pkl"
        self.similarity_engine.save_model(model_path)
        
    def _save_processed_data(self, processed_df: pd.DataFrame) -> None:
        """Save processed document vectors"""
        processed_path = self.processed_dir / "processed_documents.pkl"
        with open(processed_path, 'wb') as f:
            pickle.dump({
                'vectors': self.document_vectors,
                'indices': self.document_indices,
                'processed_df': processed_df
            }, f)
            
    def load_model(self) -> bool:
        """Load saved model and processed data"""
        model_path = self.model_dir / "similarity_model.pkl"
        processed_path = self.processed_dir / "processed_documents.pkl"
        
        if not (model_path.exists() and processed_path.exists()):
            return False
            
        # Load similarity engine
        self.similarity_engine = DocumentSimilarity.load_model(
            model_path,
            debug=self.debug
        )
        
        # Load processed data
        with open(processed_path, 'rb') as f:
            data = pickle.load(f)
            self.document_vectors = data['vectors']
            self.document_indices = data['indices']
            
        return True
        
    def find_similar(self, query: str, top_k: int = 5) -> Tuple[List[Any], List[float]]:
        """Find similar documents for a query"""
        if not self.similarity_engine:
            raise RuntimeError("Model not loaded or trained")
            
        query_vector = self.similarity_engine.transform_query(query)
        return self.similarity_engine.find_similar_documents(
            query_vector,
            self.document_vectors,
            self.document_indices,
            top_k=top_k
        )
    def find_similar_filtered(self, query: str, valid_indices: set, 
                            top_k: int = 5) -> Tuple[List[Any], List[float]]:
        """Find similar documents, but only search within valid_indices"""
        if not self.similarity_engine:
            raise RuntimeError("Model not loaded or trained")
        
        # Filter vectors to only include valid indices
        filtered_vectors, filtered_indices = self.similarity_engine.filter_document_vectors(
            self.document_vectors,
            self.document_indices,
            valid_indices
        )
        
        # If no valid documents remain after filtering
        if len(filtered_indices) == 0:
            return [], []
        
        # Perform similarity search on filtered subset
        query_vector = self.similarity_engine.transform_query(query)
        return self.similarity_engine.find_similar_documents(
            query_vector,
            filtered_vectors,
            filtered_indices,
            top_k=min(top_k, len(filtered_indices))
        )    