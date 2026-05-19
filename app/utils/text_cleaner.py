import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

MBTI_TYPES = [
    'intj', 'intp', 'entj', 'entp',
    'infj', 'infp', 'enfj', 'enfp',
    'istj', 'isfj', 'estj', 'esfj',
    'istp', 'isfp', 'estp', 'esfp'
]

MBTI_VOCAB = [
    'mbti', 'myers', 'briggs', 'myers-briggs',
    'introvert', 'introverted', 'introversion',
    'extrovert', 'extroverted', 'extroversion',
    'extravert', 'extraverted', 'extraversion',
    'intuitive', 'intuition', 'intuiting',
    'sensing', 'sensor', 'thinking', 'thinker',
    'feeling', 'feeler', 'judging', 'judger',
    'perceiving', 'perceiver', 'personality', 'type'
]

LEAKAGE_PATTERN = re.compile(
    r'\b(' + '|'.join(MBTI_TYPES + MBTI_VOCAB) + r')\b',
    re.IGNORECASE
)

# Idiomas soportados como entrada
SUPPORTED_LANGUAGES = {
    'es': 'Español',
    'en': 'Inglés'
}


def detect_language(text: str) -> str:
    """
    Detecta el idioma del texto.
    Retorna el código ISO (ej: 'es', 'en').
    Si no puede detectarlo, asume inglés.
    """
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return 'en'


def translate_to_english(text: str, source_lang: str) -> str:
    """
    Traduce el texto al inglés si no está en inglés.
    Usa GoogleTranslator de deep-translator (sin API key).
    """
    if source_lang == 'en':
        return text

    try:
        # GoogleTranslator tiene límite de 5000 caracteres por llamada
        # Si el texto es más largo lo dividimos en chunks
        max_chars = 4500
        if len(text) <= max_chars:
            translated = GoogleTranslator(
                source=source_lang,
                target='en'
            ).translate(text)
            return translated if translated else text

        # Dividir en chunks y traducir por partes
        chunks = [
            text[i:i + max_chars]
            for i in range(0, len(text), max_chars)
        ]
        translated_chunks = []
        for chunk in chunks:
            result = GoogleTranslator(
                source=source_lang,
                target='en'
            ).translate(chunk)
            translated_chunks.append(result if result else chunk)

        return ' '.join(translated_chunks)

    except Exception:
        # Si falla la traducción devolvemos el texto original
        return text


def preprocess_text(text: str) -> tuple[str, str, str]:
    """
    Pipeline completo:
    1. Detecta el idioma
    2. Traduce al inglés si es necesario
    3. Aplica limpieza y normalización

    Retorna una tupla:
        (texto_procesado, idioma_detectado, texto_traducido)
    Compatible con Python 3.12 / nltk 3.8.1 / scikit-learn 1.6.1
    """
    # Paso 0: separar posts por ||| y unir
    text = ' '.join(text.split('|||'))


    idioma_detectado = detect_language(text)

    # Paso 2: traducir al inglés si es español
    if idioma_detectado == 'es':
        text = translate_to_english(text, 'es')
    elif idioma_detectado not in SUPPORTED_LANGUAGES:
        idioma_detectado = 'en'

    texto_traducido = text

    text = text.lower()

    text = re.sub(r'http\S+|www\.\S+|https\S+', '', text)

    text = LEAKAGE_PATTERN.sub('', text)

    text = re.sub(r'[^a-z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)

    tokens = [
        t for t in tokens
        if t not in STOP_WORDS and len(t) > 2
    ]

    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return ' '.join(tokens), idioma_detectado, texto_traducido


def preprocess_text(text: str) -> str:
    """
    Pipeline de preprocesamiento idéntico al usado
    durante el entrenamiento del modelo.
    Compatible con Python 3.12 / nltk 3.8.1 / scikit-learn 1.6.1
    """
   
    text = ' '.join(text.split('|||'))


    text = text.lower()

    text = re.sub(r'http\S+|www\.\S+|https\S+', '', text)

    text = LEAKAGE_PATTERN.sub('', text)

    text = re.sub(r'[^a-z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)

    tokens = [
        t for t in tokens
        if t not in STOP_WORDS and len(t) > 2
    ]

    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return ' '.join(tokens)