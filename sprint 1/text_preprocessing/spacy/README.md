# Text-Preprocessing with Spacy

spaCy is a free, open-source library for advanced Natural Language Processing (NLP) in Python.

## Installation

Before beginning, ensure you have spaCy and a language model installed. You can install spaCy and download the English language model using pip:

```bash
pip install -U spacy
python -m spacy download en_core_web_sm
```

# Usage

## Import spaCy and download model

```bash
import spacy
nlp = spacy.load("en_core_web_sm")
```

## Creating Doc container

The doc object contains a sequence of tokens, each of which represents a word or a punctuation mark in the document. The Doc object also stores various linguistic annotations and information about the text, such as part-of-speech tags, named entities, dependency parse information, and more.

```bash
text = "this is a sample text"
doc = nlp(text)
```

## Tokenization

### Tokenize Sentences:

Breaking a text into individual sentences.

```bash
def tokenize_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]
```

### Tokenize Words:

Splitting a text into individual words or tokens.

```bash
def tokenize_words(text):
    doc = nlp(text)
    return [token.text for token in doc]
```

## Lowercasing

Converts input text to lowercase.

```bash
def lowercase(text):
    text_to_lower = text.lower()
    return text_to_lower
```

## Remove stopwords

Removes stopwords from input text. Stopwords are common words (e.g., "the," "and," "is") that are often removed from text because they do not carry significant meaning.

```bash
def remove_stopwords(text):
    doc = nlp(text)
    filtered_text = " ".join(token.text for token in doc if not token.is_stop)
    return filtered_text
```

## Lemmatization

Lemmatization is the process of reducing words to their dictionary or base form (lemma).

```bash
def lemmatize_word(text):
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc]
    return lemmatized_words
```

## Named Entity Recognition

Named Entity Recognition identifies and categorizes named entities (e.g., names of people, places, organizations) in text.

```bash
def named_entity_recognition(text):
    """
    Extracts named entities from input text.

    Args:
        text (str): The input text.

    Returns:
        spacy.tokens.Doc: A spaCy Doc object containing named entities.
    """
    doc = nlp(text)
    return doc.ents

named_entities = named_entity_recognition(text)
print("\nNamed Entities:")
for entity in named_entities:
    print(f"Text: {entity.text}, Label: {entity.label_}")
```