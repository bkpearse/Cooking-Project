"""Core logic powering the Project Cook interfaces.

This module exposes small, dependency-light helpers that can be reused by the
FastAPI service, the Streamlit front-end or any other integration.  The
previous implementation eagerly imported heavy optional dependencies and
expected local assets (model weights, datasets and indices) to always be
available.  That behaviour made deployments brittle.  The refactor below keeps
those dependencies optional, performs lazy loading and provides helpful error
messages so the application can run in a variety of environments (including a
fresh Streamlit deployment).
"""

from __future__ import annotations

import csv
import json
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
from PIL import Image
from whoosh import index as whoosh_index
from whoosh.fields import ID, KEYWORD, Schema, TEXT
from whoosh.qparser import QueryParser

from project_cook.logic.clean_text import basic_cleaning, remove_punctuation, remove_words

try:  # The translator is entirely optional.
    from googletrans import Translator as _GoogleTranslator
except ImportError:  # pragma: no cover - dependency not installed by default.
    _GoogleTranslator = None  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_SIZE = int(os.getenv("COOK_IMAGE_SIZE", "224"))
MODEL_PATH = Path(os.getenv("COOK_MODEL_PATH", "notebooks/model.h5"))
LABELS_PATH_RAW = os.getenv("COOK_LABELS_PATH", "")
LABELS_PATH = Path(LABELS_PATH_RAW) if LABELS_PATH_RAW else None
TRAIN_IMAGE_DIR = Path(os.getenv("COOK_TRAIN_IMAGE_DIR", "notebooks/images/train"))
INDEX_DIR = Path(os.getenv("COOK_INDEX_DIR", "new_index"))
DATASET_PATH = Path(
    os.getenv("COOK_DATASET_PATH", "project_cook/data/full_dataset.csv"))
INDEX_CHUNK_SIZE = int(os.getenv("COOK_INDEX_CHUNK_SIZE", "10000"))

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_label_lookup() -> Dict[int, str]:
    """Return a mapping between numeric class indices and human readable labels.

    The mapping is inferred in three different ways (in order of priority):

    1. From a JSON file referenced via ``COOK_LABELS_PATH``.  This allows
       deployments to ship a simple ``["class_a", "class_b"]`` payload
       alongside the model.
    2. From the ``notebooks/images/train`` directory (or the directory pointed
       to by ``COOK_TRAIN_IMAGE_DIR``) by scanning the folder names.  This
       mirrors ``ImageDataGenerator``'s behaviour which sorts directory names.
    3. When neither of the above are available, the mapping is empty and we fall
       back to synthetic ``class_<index>`` labels.
    """

    if LABELS_PATH and LABELS_PATH.exists():
        try:
            payload = json.loads(LABELS_PATH.read_text())
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(payload, dict):
                lookup: Dict[int, str] = {}
                for key, value in payload.items():
                    try:
                        lookup[int(key)] = str(value)
                    except ValueError:
                        continue
                if lookup:
                    return lookup
            if isinstance(payload, list):
                return {idx: str(label) for idx, label in enumerate(payload)}

    if TRAIN_IMAGE_DIR.exists():
        class_names = sorted(
            entry.name for entry in TRAIN_IMAGE_DIR.iterdir() if entry.is_dir())
        if class_names:
            return {idx: name for idx, name in enumerate(class_names)}

    return {}


@lru_cache(maxsize=1)
def _get_translator() -> Optional[_GoogleTranslator]:
    if _GoogleTranslator is None:  # pragma: no cover - optional dependency
        return None
    return _GoogleTranslator()


def _iter_dataset_rows(dataset_path: Path) -> Iterator[Dict[str, str]]:
    """Yield rows from the CSV dataset as dictionaries.

    The historical dataset ships with an initial identifier column, therefore we
    gracefully handle both ``DictReader`` friendly CSV files (with explicit
    headers) and raw files without headers.
    """

    if not dataset_path.exists():
        return

    with dataset_path.open(newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader, None)

    if headers and {'title', 'ingredients', 'directions'}.issubset(set(headers)):
        with dataset_path.open(newline='', encoding='utf-8') as file:
            keyed_reader = csv.DictReader(file)
            for row in keyed_reader:
                yield {
                    'title': row.get('title', ''),
                    'ingredients': row.get('ingredients', ''),
                    'directions': row.get('directions', ''),
                    'link': row.get('link', ''),
                    'source': row.get('source', ''),
                    'NER': row.get('NER', ''),
                }
        return

    # Fall back to manual positional extraction using the previously read header.
    if headers is not None and len(headers) >= 6:
        yield {
            'title': headers[1] if len(headers) > 1 else '',
            'ingredients': headers[2] if len(headers) > 2 else '',
            'directions': headers[3] if len(headers) > 3 else '',
            'link': headers[4] if len(headers) > 4 else '',
            'source': headers[5] if len(headers) > 5 else '',
            'NER': headers[6] if len(headers) > 6 else '',
        }

    with dataset_path.open(newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip the first row that we've already processed.
        for row in reader:
            if len(row) < 6:
                continue
            yield {
                'title': row[1] if len(row) > 1 else '',
                'ingredients': row[2] if len(row) > 2 else '',
                'directions': row[3] if len(row) > 3 else '',
                'link': row[4] if len(row) > 4 else '',
                'source': row[5] if len(row) > 5 else '',
                'NER': row[6] if len(row) > 6 else '',
            }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_model():
    """Load and compile the TensorFlow model lazily.

    The function caches the model instance to avoid repeated disk reads when the
    Streamlit app receives multiple requests.  A ``RuntimeError`` with a helpful
    message is raised if TensorFlow is not installed or if the model file is
    missing.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not locate the model at '{MODEL_PATH}'. Set COOK_MODEL_PATH to the"
            " correct location before making predictions.")

    try:  # Lazy import to avoid importing TensorFlow during Streamlit boot.
        from tensorflow.keras import optimizers
        from tensorflow.keras.models import load_model
    except ImportError as exc:  # pragma: no cover - heavy optional dependency
        raise RuntimeError(
            "TensorFlow is required to load the cooking model. Install it by"
            " adding 'tensorflow' to your requirements or by running 'pip install"
            " tensorflow'.") from exc

    model = load_model(str(MODEL_PATH), compile=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=1e-4),
                  metrics=['accuracy'])
    return model


def pred_streamlit(user_input: bytes, lang: str) -> Dict[str, List[str]]:
    """Generate predictions for an uploaded image.

    Parameters
    ----------
    user_input:
        Raw bytes coming from an uploaded image.
    lang:
        Target language code.  When translation is not available the original
        English labels are returned.
    """

    try:  # Lazy import – keeps Streamlit's initial footprint minimal.
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        from tensorflow.keras.preprocessing.image import img_to_array
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "TensorFlow is required to run image predictions. Install it via"
            " pip or ensure it is present in your environment.") from exc

    image = Image.open(BytesIO(user_input)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    model = get_model()
    prediction_scores = model.predict(image_array)
    predicted_probabilities = np.argmax(prediction_scores, axis=1)

    labels = _load_label_lookup()
    predictions = [labels.get(idx, f"class_{idx}") for idx in predicted_probabilities]

    if lang == 'en':
        translations = predictions
    else:
        translations = [translate_text(label, lang) for label in predictions]

    return {'prediction': predictions, 'translation': translations}


def settings(filename: str | os.PathLike[str] = DATASET_PATH
             ) -> Optional[whoosh_index.Index]:
    """Return (and build when necessary) the search index used by Whoosh."""

    dataset_path = Path(filename)
    if not dataset_path.exists():
        return None

    if whoosh_index.exists_in(str(INDEX_DIR)):
        return whoosh_index.open_dir(str(INDEX_DIR))

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    schema = Schema(title=TEXT(stored=True),
                    ingredients=KEYWORD(stored=True, commas=True),
                    directions=TEXT(stored=True),
                    link=ID(stored=True),
                    source=TEXT(stored=True),
                    NER=TEXT(stored=True))

    ix = whoosh_index.create_in(str(INDEX_DIR), schema)
    writer = ix.writer(limitmb=256)

    buffer: List[Dict[str, str]] = []
    for row in _iter_dataset_rows(dataset_path):
        buffer.append(row)
        if len(buffer) >= INDEX_CHUNK_SIZE:
            for entry in buffer:
                writer.add_document(**entry)
            buffer = []

    for entry in buffer:
        writer.add_document(**entry)
    writer.commit()
    return ix


def search_recipes(search_term: str, lang: str) -> List[Dict[str, str]]:
    """Search the indexed recipes using a user provided query."""

    cleaned_term = remove_words(remove_punctuation(basic_cleaning(search_term)))

    ix = settings()
    if ix is None:
        return []

    qp = QueryParser("NER", schema=ix.schema)
    q = qp.parse(cleaned_term)

    recipes: List[Dict[str, str]] = []
    with ix.searcher() as searcher:
        results = searcher.search(q, limit=None)
        for result in results:
            hit = {
                'NER': result['NER'],
                'directions': result['directions'],
                'ingredients': result['ingredients'],
                'link': result['link'],
                'source': result['source'],
                'title': result['title'],
            }
            if lang != 'en':
                hit = {
                    'NER': translate_text(hit['NER'], lang),
                    'directions': translate_text(hit['directions'], lang),
                    'ingredients': translate_text(hit['ingredients'], lang),
                    'link': hit['link'],
                    'source': hit['source'],
                    'title': translate_text(hit['title'], lang),
                }
            recipes.append(hit)
    return recipes


def translate_text(text: str, target_language: str = 'en') -> str:
    """Translate ``text`` into ``target_language`` when possible."""

    if target_language == 'en':
        return text

    translator = _get_translator()
    if translator is None:  # pragma: no cover - optional dependency
        return text

    try:
        translation = translator.translate(text, dest=target_language)
    except Exception:  # pragma: no cover - API/network errors should not crash the app
        return text
    return translation.text


__all__ = ['get_model', 'pred_streamlit', 'search_recipes', 'settings', 'translate_text']
