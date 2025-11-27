from deepface import DeepFace
from src.config import FACE_DETECTION_MODEL

class FaceDetector:
    @staticmethod
    def detect_faces(img_path):
        """Retorna lista de faces encontradas na imagem"""
        try:
            return DeepFace.extract_faces(
                img_path=img_path,
                detector_backend=FACE_DETECTION_MODEL,
                enforce_detection=False,
                align=True
            )
        except Exception as e:
            print(f"Erro ao detectar face: {e}")
            return []