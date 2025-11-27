import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Caminhos do projeto
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
DATABASE_DIR = DATA_DIR / "database"

# Cria pastas se não existirem
for dir_path in [INPUT_DIR, OUTPUT_DIR, DATABASE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Banco de Dados
DATABASE_URL = f"sqlite:///{DATABASE_DIR}/faces.db"

# Modelos de IA (O coração do sistema)
# RetinaFace é mais lento, mas MUITO preciso (acha rosto até de máscara ou perfil)
FACE_DETECTION_MODEL = "retinaface"
# ArcFace é o estado da arte pra transformar rosto em números
FACE_RECOGNITION_MODEL = "ArcFace"