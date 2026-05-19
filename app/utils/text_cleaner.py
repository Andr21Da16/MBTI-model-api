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

SUPPORTED_LANGUAGES = {
    'es': 'Español',
    'en': 'Inglés'
}


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return 'en'


def translate_to_english(text: str, source_lang: str) -> str:
    if source_lang == 'en':
        return text
    try:
        max_chars = 4500
        if len(text) <= max_chars:
            translated = GoogleTranslator(
                source=source_lang,
                target='en'
            ).translate(text)
            return translated if translated else text

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
        return text


def preprocess_text(text: str):
    """
    Retorna exactamente una tupla de 3 elementos:
    (texto_procesado, idioma_detectado, texto_traducido)
    """
    # Paso 0: separar posts por ||| y unir
    text = ' '.join(text.split('|||'))


    idioma_detectado = detect_language(text)


    if idioma_detectado == 'es':
        text = translate_to_english(text, 'es')
    elif idioma_detectado not in SUPPORTED_LANGUAGES:
        idioma_detectado = 'en'

   
    texto_traducido = str(text)

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

    texto_final = ' '.join(tokens)
    return texto_final, idioma_detectado, texto_traducido