import os
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from app.utils.text_cleaner import (
    preprocess_text,
    SUPPORTED_LANGUAGES
)
from download_model import download_model

load_dotenv()
MODEL_PATH  = os.getenv('MODEL_PATH', 'model/mbti_pipeline.pkl')
APP_VERSION = os.getenv('APP_VERSION', '1.0.0')

model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    1. Descarga el modelo si no existe localmente
    2. Lo carga en memoria una sola vez al iniciar
    """
    download_model()

    try:
        pipeline = joblib.load(MODEL_PATH)
        model_state['pipeline']    = pipeline
        model_state['clases']      = list(
            pipeline.named_steps['clf'].classes_
        )
        model_state['tiene_proba'] = hasattr(
            pipeline.named_steps['clf'], 'predict_proba'
        )
        print(f"Modelo cargado: {MODEL_PATH}")
        print(f"Clases: {model_state['clases']}")
    except Exception as e:
        raise RuntimeError(
            f"No se pudo cargar el modelo: {e}"
        )
    yield
    model_state.clear()
    print("Servidor apagado. Modelo liberado.")



app = FastAPI(
    title="MBTI Personality Predictor API",
    description=(
        "Predice el tipo de personalidad MBTI a partir "
        "de texto escrito por el usuario. "
        "Soporta entrada en Español e Inglés. "
        "Modelo entrenado con dataset de PersonalityCafe (Kaggle)."
    ),
    version=APP_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



class PredictRequest(BaseModel):
    text: str

    @field_validator('text')
    @classmethod
    def validar_texto(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                'El campo text no puede estar vacío.'
            )
        if len(v.strip().split()) < 10:
            raise ValueError(
                'Texto demasiado corto. '
                'Proporciona al menos 10 palabras.'
            )
        return v.strip()


class AxisPrediction(BaseModel):
    I_E: str
    N_S: str
    T_F: str
    J_P: str


class PredictResponse(BaseModel):
    mbti_type:        str
    axis_predictions: AxisPrediction
    probabilities:    dict | None = None
    detected_language: str
    translated:       bool
    warning:          str | None  = None


@app.get(
    "/",
    tags=["Health"],
    summary="Estado general de la API"
)
def root():
    return {
        "status":             "ok",
        "version":            APP_VERSION,
        "idiomas_soportados": SUPPORTED_LANGUAGES,
        "docs":               "/docs"
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Verifica que el modelo está cargado"
)
def health():
    return {
        "status":         "ok",
        "modelo_cargado": "pipeline" in model_state,
        "clases":         model_state.get("clases", []),
        "total_clases":   len(model_state.get("clases", []))
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Predicción"],
    summary="Predice el tipo MBTI a partir de texto en ES o EN"
)
def predict(request: PredictRequest):

    texto_limpio, idioma, texto_traducido = preprocess_text(
        request.text
    )

    if len(texto_limpio.strip().split()) < 5:
        raise HTTPException(
            status_code=422,
            detail=(
                "El texto quedó muy corto tras el "
                "preprocesamiento. Intenta con más contenido."
            )
        )

    try:
        pipeline = model_state['pipeline']

        tipo_predicho = pipeline.predict([texto_limpio])[0]

        ejes = AxisPrediction(
            I_E=tipo_predicho[0],
            N_S=tipo_predicho[1],
            T_F=tipo_predicho[2],
            J_P=tipo_predicho[3]
        )

        probabilidades = None
        if model_state['tiene_proba']:
            proba  = pipeline.predict_proba([texto_limpio])[0]
            clases = model_state['clases']
            probabilidades = dict(
                sorted(
                    {k: round(float(v), 4)
                     for k, v in zip(clases, proba)}.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            )

        warning = None
        if tipo_predicho[0] == 'I':
            warning = (
                "El modelo presenta tendencia hacia tipos "
                "Introvertidos (ratio I/E ~3.3x en el dataset). "
                "Interpreta el resultado como una estimación "
                "probabilística."
            )

        return PredictResponse(
            mbti_type=tipo_predicho,
            axis_predictions=ejes,
            probabilities=probabilidades,
            detected_language=SUPPORTED_LANGUAGES.get(
                idioma, f'No soportado ({idioma})'
            ),
            translated=idioma != 'en',
            warning=warning
        )

    except KeyError:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno: {str(e)}"
        )