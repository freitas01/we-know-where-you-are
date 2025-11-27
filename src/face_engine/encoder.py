from deepface import DeepFace
from src.config import FACE_RECOGNITION_MODEL


class FaceEncoder:
    """Gera a assinatura digital (hash) do rosto"""

    @staticmethod
    def generate_embedding(img_path):
        try:
            # O DeepFace.represent gera o vetor matemático do rosto
            embedding_objs = DeepFace.represent(
                img_path=img_path,
                model_name=FACE_RECOGNITION_MODEL,
                enforce_detection=False
            )

            # Retorna apenas os vetores (embeddings)
            return [obj["embedding"] for obj in embedding_objs]
        except Exception as e:
            print(f"❌ Erro ao gerar embedding: {e}")
            return []