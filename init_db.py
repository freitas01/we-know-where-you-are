import logging
import sys
from src.database.repository import Repository

# Configuração de Logs / Log Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def initialize_database():
    """
    Initializes the database schema.
    Inicializa o esquema do banco de dados.
    """
    logger.info("Starting database setup... | Iniciando configuração do banco de dados...")

    try:
        # Repository instantiation invokes SQLAlchemy's create_all
        repo = Repository()
        stats = repo.get_stats()

        logger.info("Database initialized successfully. | Banco de dados inicializado com sucesso.")
        logger.info(f"Location: data/database/faces.db")

        logger.info("-" * 50)
        logger.info("SYSTEM INITIAL STATISTICS | ESTATÍSTICAS INICIAIS DO SISTEMA")
        logger.info("-" * 50)
        logger.info(f"[-] Registered Persons | Pessoas Registradas: {stats['total_persons']}")
        logger.info(f"[-] Vectorized Faces   | Faces Vetorizadas:   {stats['total_faces']}")
        logger.info(f"[-] Processing Logs    | Logs de Processamento: {stats['total_processed']}")
        logger.info("-" * 50)

    except Exception as e:
        logger.error(f"Critical failure during database initialization | Falha crítica na inicialização: {e}")
        sys.exit(1)


if __name__ == "__main__":
    initialize_database()