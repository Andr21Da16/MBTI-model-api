import os
import gdown
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR      = 'model'
MODEL_PATH     = os.path.join(MODEL_DIR, 'mbti_pipeline.pkl')
GDRIVE_FILE_ID = os.getenv(
    'GDRIVE_FILE_ID',
    '12KmyaF_M3g2oHr9fDse7jwcyDL1osAhs'
)
GDRIVE_URL = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'


def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"Modelo ya existe en: {MODEL_PATH}")
        return

    print("Modelo no encontrado. Descargando desde Google Drive...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"Modelo descargado correctamente ({size_mb:.2f} MB)")
    else:
        raise RuntimeError(
            "La descarga falló. Verifica que el archivo "
            "de Drive es público."
        )


if __name__ == '__main__':
    download_model()