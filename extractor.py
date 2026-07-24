"""Extract normalized subject--relation--object triples with spaCy."""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import nltk
import spacy
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from settings import NLTK_DATA_PATH


nltk.data.path.insert(0, str(NLTK_DATA_PATH))
wordnet_lemmatizer = WordNetLemmatizer()


class Relationship(NamedTuple):
    subject: str
    relation: str
    object: str


@lru_cache(maxsize=1)
def get_nlp():
    """Load the small English pipeline with a helpful setup error."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError as error:
        raise RuntimeError(
            "Missing spaCy English model. Run: python -m spacy download en_core_web_sm"
        ) from error


def _phrase(token) -> str:
    """Use a compound phrase (e.g. 'carbon dioxide') as one graph concept."""
    words = [child.lemma_.lower() for child in token.lefts if child.dep_ in {"compound", "amod"}]
    words.append(token.lemma_.lower())
    return " ".join(word for word in words if word and word != "-pron-")


def _subjects(verb):
    subjects = [child for child in verb.children if child.dep_ in {"nsubj", "nsubjpass", "csubj"}]
    if subjects:
        return subjects
    # Coordinated verbs often inherit the subject from the governing verb.
    if verb.dep_ in {"conj", "xcomp"} and verb.head.pos_ in {"VERB", "AUX"}:
        return _subjects(verb.head)
    return []


def _objects(verb):
    return [child for child in verb.children if child.dep_ in {"dobj", "obj", "attr", "oprd"}]


def _fallback_relationship(sentence) -> Relationship | None:
    """Recover compact SVO claims when the statistical parser sees a noun phrase.

    Small English models sometimes parse valid short claims such as "Memory
    stores data" as three nouns. WordNet lets us safely identify a verb-like
    middle token without applying this heuristic to ordinary noun phrases.
    """
    tokens = [
        token
        for token in sentence
        if token.pos_ not in {"DET", "PUNCT", "SPACE"} and not token.is_punct
    ]
    if len(tokens) < 3:
        return None
    for position in range(1, len(tokens) - 1):
        candidate = tokens[position]
        lemma = candidate.lemma_.lower()
        try:
            is_verb = bool(wordnet.synsets(lemma, pos=wordnet.VERB))
        except LookupError:
            is_verb = False
        if not is_verb:
            continue
        subject = " ".join(
            wordnet_lemmatizer.lemmatize(token.text.lower(), "n")
            for token in tokens[:position]
        )
        object_ = " ".join(
            wordnet_lemmatizer.lemmatize(token.text.lower(), "n")
            for token in tokens[position + 1 :]
        )
        if subject and object_:
            return Relationship(subject, lemma, object_)
    return None


def extract_relationships(text: str) -> list[Relationship]:
    """Return unique, lemmatized relations expressed in *text*.

    The extractor intentionally captures only explicit subject-verb-object claims.
    This makes results explainable and avoids inventing connections from keywords.
    """
    doc = get_nlp()(text)
    relationships: set[Relationship] = set()

    for sentence in doc.sents:
        count_before = len(relationships)
        for verb in sentence:
            if verb.pos_ not in {"VERB", "AUX"}:
                continue
            subjects, objects = _subjects(verb), _objects(verb)
            for subject in subjects:
                for obj in objects:
                    source, target = _phrase(subject), _phrase(obj)
                    relation = verb.lemma_.lower()
                    if any(child.dep_ == "neg" for child in verb.children):
                        relation = f"not_{relation}"
                    if source and target and relation:
                        relationships.add(Relationship(source, relation, target))
        if len(relationships) == count_before:
            fallback = _fallback_relationship(sentence)
            if fallback is not None:
                relationships.add(fallback)

    return sorted(relationships)
