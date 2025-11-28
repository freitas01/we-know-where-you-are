from deepface import DeepFace
from src.config import FACE_RECOGNITION_MODEL


class FaceEncoder:
    """Generates facial embeddings (mathematical signature of a face)"""
    
    @staticmethod
    def generate_embedding(img_path: str) -> list:
        """
        Generate embedding vector for faces in image
        Returns list of embedding vectors (one per face found)
        """
        try:
            embedding_objs = DeepFace.represent(
                img_path=img_path,
                model_name=FACE_RECOGNITION_MODEL,
                enforce_detection=False
            )
            return [obj["embedding"] for obj in embedding_objs]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
