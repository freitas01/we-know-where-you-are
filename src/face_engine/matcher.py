"""Face matching module - compares embeddings to find known persons"""

import numpy as np
from scipy.spatial.distance import cosine
from typing import Optional, Tuple
from src.database.repository import Repository
from src.config import FACE_DISTANCE_THRESHOLD


class FaceMatcher:
    """Compares face embeddings to identify known persons"""

    def __init__(self):
        self.repo = Repository()
        self.threshold = FACE_DISTANCE_THRESHOLD

    def _bytes_to_embedding(self, embedding_bytes: bytes) -> np.ndarray:
        """Convert stored bytes back to numpy array"""
        return np.frombuffer(embedding_bytes, dtype=np.float32)

    def _embedding_to_bytes(self, embedding: list) -> bytes:
        """Convert embedding list to bytes for storage"""
        return np.array(embedding, dtype=np.float32).tobytes()

    def calculate_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine distance between two embeddings
        Returns: 0.0 = identical, 1.0 = completely different
        """
        return cosine(embedding1, embedding2)

    def find_match(self, new_embedding: list) -> Optional[Tuple[int, float]]:
        """
        Search database for matching face

        Args:
            new_embedding: List of floats (512 dimensions from ArcFace)

        Returns:
            Tuple (person_id, distance) if match found, None otherwise
        """
        new_vector = np.array(new_embedding, dtype=np.float32)

        # Get all stored embeddings
        stored_faces = self.repo.get_all_face_embeddings()

        if not stored_faces:
            return None

        best_match = None
        best_distance = float('inf')

        for person_id, embedding_bytes in stored_faces:
            stored_vector = self._bytes_to_embedding(embedding_bytes)
            distance = self.calculate_distance(new_vector, stored_vector)

            if distance < best_distance:
                best_distance = distance
                best_match = person_id

        # Check if best match is within threshold
        if best_distance < self.threshold:
            return (best_match, best_distance)

        return None

    def is_same_person(self, embedding1: list, embedding2: list) -> Tuple[bool, float]:
        """
        Compare two embeddings directly

        Returns:
            Tuple (is_match: bool, distance: float)
        """
        vec1 = np.array(embedding1, dtype=np.float32)
        vec2 = np.array(embedding2, dtype=np.float32)
        distance = self.calculate_distance(vec1, vec2)

        return (distance < self.threshold, distance)

    def get_all_matches(self, new_embedding: list, top_k: int = 5) -> list:
        """
        Get top K closest matches from database

        Returns:
            List of tuples [(person_id, distance), ...] sorted by distance
        """
        new_vector = np.array(new_embedding, dtype=np.float32)
        stored_faces = self.repo.get_all_face_embeddings()

        if not stored_faces:
            return []

        matches = []
        for person_id, embedding_bytes in stored_faces:
            stored_vector = self._bytes_to_embedding(embedding_bytes)
            distance = self.calculate_distance(new_vector, stored_vector)
            matches.append((person_id, distance))

        # Sort by distance (closest first)
        matches.sort(key=lambda x: x[1])

        return matches[:top_k]