import os
import cv2
from src.face_engine.detector import FaceDetector
from src.face_engine.encoder import FaceEncoder

# Pega a primeira imagem que achar na pasta input
input_folder = "data/input"
files = os.listdir(input_folder)
if not files:
    print(f"⚠️ Coloque uma foto em {input_folder} para testar!")
    exit()

img_name = files[0]
img_path = os.path.join(input_folder, img_name)

print(f"\n📸 PROCESSANDO: {img_name}")
print("-" * 30)

# 1. Detectar
print("🔍 Procurando faces...")
faces = FaceDetector.detect_faces(img_path)
print(f"✅ Faces encontradas: {len(faces)}")

if len(faces) > 0:
    # 2. Gerar Embedding
    print("🔢 Gerando assinatura digital...")
    embeddings = FaceEncoder.generate_embedding(img_path)

    # Mostra resultado
    print("-" * 30)
    print(f"🚀 SUCESSO! O sistema 'We Know Where You Are' está vivo.")
    print(f"📊 Dimensões do vetor facial: {len(embeddings[0])}")
    print(f"🧬 Assinatura (primeiros 5 números): {embeddings[0][:5]}...")

    # (Opcional) Mostra a imagem com retângulo se tiver interface gráfica
    # img = cv2.imread(img_path)
    # cv2.imshow("Teste", img)
    # cv2.waitKey(0)
else:
    print("❌ Nenhuma face detectada.")