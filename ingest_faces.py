import os
import logging
import numpy as np
from src.face_engine.detector import FaceDetector
from src.face_engine.encoder import FaceEncoder
from src.database.repository import Repository
from src.config import INPUT_DIR

# Configuração de Logs / Log Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_input_directory():
    """
    Processes images from input directory.
    Processa imagens do diretório de entrada.
    """
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        logger.error(f"Input directory not found: {INPUT_DIR} | Diretório de entrada não encontrado.")
        return

    # Filter for valid image extensions
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not files:
        logger.warning("No valid images found for processing. | Nenhuma imagem válida encontrada.")
        return

    repo = Repository()
    logger.info(
        f"Starting batch processing: {len(files)} files. | Iniciando processamento em lote: {len(files)} arquivos.")

    for filename in files:
        file_path = os.path.join(INPUT_DIR, filename)
        logger.info("-" * 30)
        logger.info(f"Processing file: {filename} | Processando arquivo: {filename}")

        try:
            # Step 1: Face Detection
            faces = FaceDetector.detect_faces(file_path)

            if not faces:
                logger.warning(f" [!] No faces detected in {filename} | Nenhuma face detectada.")
                continue

            logger.info(f" [i] Faces detected: {len(faces)} | Faces detectadas: {len(faces)}")

            # Step 2: Feature Extraction (Embedding)
            embeddings = FaceEncoder.generate_embedding(file_path)

            # Step 3: Persistence
            for i, embedding_vector in enumerate(embeddings):
                # Technical conversion: Float List -> Numpy Array -> Bytes
                vector_bytes = np.array(embedding_vector, dtype=np.float32).tobytes()

                # Entity Creation (Person) - DADOS EM INGLÊS NO BANCO
                person_name = f"Subject_{filename}_{i}"
                person = repo.create_person(name=person_name)

                # Persist Face (Vector)
                repo.add_face_to_person(
                    person_id=person.id,
                    embedding=vector_bytes,
                    confidence=0.99,
                    source_image=filename
                )

                # Tracking Record (Sighting)
                repo.add_sighting(
                    person_id=person.id,
                    source_type="batch_ingestion",  # DATA IN ENGLISH
                    source_file=filename,
                    captured_at=None
                )

                logger.info(f"   [+] Persisted: ID {person.id} (UUID: {person.unique_id}) | Salvo no Banco")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e} | Erro ao processar arquivo.")

    # Operation Summary
    stats = repo.get_stats()
    logger.info("=" * 50)
    logger.info("PROCESSING COMPLETED | PROCESSAMENTO CONCLUÍDO")
    logger.info(f"Total Persons in Database | Total de Pessoas no Banco: {stats['total_persons']}")
    logger.info("=" * 50)


if __name__ == "__main__":
    process_input_directory()