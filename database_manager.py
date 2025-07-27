import json
import numpy as np
from typing import Optional, Tuple, List
import os

class DatabaseManager:
    def __init__(self, json_path):
        self.json_path = json_path
        self.data = {}
        self.load_database()
    
    def load_database(self):
        """Load the face embeddings from JSON file (silent)"""
        try:
            if os.path.exists(self.json_path):
                with open(self.json_path, 'r') as f:
                    self.data = json.load(f)
            else:
                self.data = {}
        except:
            self.data = {}
    
    def find_similar_face(self, query_embedding: np.ndarray, threshold: float = 0.60) -> Optional[Tuple[str, str, float]]:
        """Find similar face in JSON database (silent)"""
        try:
            if not self.data:
                return None
            
            best_match = None
            best_similarity = 0
            
            for person_name, person_data in self.data.items():
                stored_embedding = np.array(person_data["mean_embedding"])
                similarity = self.cosine_similarity(query_embedding, stored_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    if similarity >= threshold:
                        best_match = (person_name, person_name, similarity)
            
            return best_match
        except:
            return None
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity (silent)"""
        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0
            
            return dot_product / (norm1 * norm2)
        except:
            return 0
    
    def get_all_persons(self) -> List[str]:
        """Get all persons (silent)"""
        try:
            return list(self.data.keys())
        except:
            return []
    
    def get_person_info(self, person_name: str):
        """Get person info (silent)"""
        if person_name in self.data:
            return self.data[person_name]
        return None