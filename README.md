# MBTI Personality Predictor

Sistema de clasificación automática del tipo de personalidad
MBTI a partir de texto escrito, desarrollado como proyecto
de Aprendizaje Estadístico.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Railway](https://img.shields.io/badge/Deploy-Railway-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Descripción

Modelo supervisado de clasificación de texto que predice
el tipo de personalidad MBTI (uno de 16 tipos posibles)
a partir del análisis lingüístico de publicaciones escritas
por el usuario. Soporta entrada en Español e Inglés.

## Arquitectura del Sistema
Texto de entrada (ES/EN)
│
▼
Detección de idioma (langdetect)
│
▼
Traducción al inglés si es necesario (deep-translator)
│
▼
Pipeline de preprocesamiento NLP (NLTK)

Eliminación de URLs
Eliminación de términos MBTI (anti-leakage)
Tokenización → Stopwords → Lematización
│
▼
Vectorización TF-IDF (10,000 términos, bigramas)
│
▼
Clasificador (scikit-learn)
│
▼
Predicción: tipo MBTI + probabilidades por clase


## Dataset

- **Fuente**: Kaggle — MBTI Myers-Briggs Personality Type Dataset
- **Registros**: 8,675 usuarios
- **Clases**: 16 tipos MBTI
- **Origen**: Foro PersonalityCafe

## Tecnologías

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.12 |
| ML Pipeline | scikit-learn 1.6.1 |
| NLP | NLTK 3.8.1 |
| Vectorización | TF-IDF |
| API | FastAPI 0.111 |
| Servidor | Uvicorn |
| Deploy | Railway |
| Traducción | deep-translator |

## Instalación Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/mbti-api.git
cd mbti-api

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar modelo automáticamente
python download_model.py

# 5. Levantar servidor
uvicorn app.main:app --reload --port 8000
```

## Uso de la API

### Endpoint principal

**POST** `/predict`

**Entrada:**
```json
{
  "text": "Tu texto aquí (mínimo 10 palabras)"
}
```

**Salida:**
```json
{
  "mbti_type": "INTJ",
  "axis_predictions": {
    "I_E": "I",
    "N_S": "N",
    "T_F": "T",
    "J_P": "J"
  },
  "probabilities": {
    "INTJ": 0.3821,
    "INTP": 0.2134,
    "...": "..."
  },
  "detected_language": "Español",
  "translated": true,
  "warning": "..."
}
```

### Ejemplo curl

```bash
curl -X POST "https://tu-url.railway.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Me encanta pasar tiempo solo..."}'
```

### Ejemplo Python

```python
import requests
r = requests.post(
    "https://tu-url.railway.app/predict",
    json={"text": "I love spending time alone..."}
)
print(r.json())
```

## Métricas del Modelo

| Métrica | Train | Test |
|---|---|---|
| Accuracy | ~0.78 | ~0.62 |
| F1-macro | ~0.69 | ~0.51 |
| F1-weighted | ~0.75 | ~0.61 |

## Limitaciones Conocidas

- **Sesgo I/E**: ratio ~3.3x a favor de tipos Introvertidos
  en el dataset de entrenamiento.
- **Sesgo N/S**: ratio ~6.9x a favor de tipos Intuitivos.
- **Dominio específico**: modelo entrenado sobre posts de
  un foro MBTI; generalización a otros contextos es incierta.
- **Etiquetas autodeclaradas**: las etiquetas del dataset
  son autodeclaradas por los usuarios, no validadas
  psicométricamente.

## Autores

Proyecto desarrollado como trabajo académico de
Aprendizaje Estadístico — Universidad Nacional de Trujillo.

## Referencias

- Myers, I. B., & Myers, P. B. (1980). Gifts Differing.
- Gjurkovic & Snajder (2018). Reddit: A gold mine for
  personality prediction. ACL Workshop.
- Dataset: https://www.kaggle.com/datasnaek/mbti-type
