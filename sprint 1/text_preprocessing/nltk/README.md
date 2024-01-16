# Text Preprocessing with NLTK

Natural Language Toolkit (NLTK) stands as a prominent platform for developing Python programs focused on working with human language data. It offers user-friendly interfaces to over 50 corpora and lexical resources such as WordNet. Alongside, it provides a suite of text processing libraries encompassing classification, tokenization, stemming, tagging, parsing, and semantic reasoning, along with wrappers for industrial-strength Natural Language Processing (NLP) libraries.

## Requirements

Before delving into the tasks, the NLTK library needs installation. This can be achieved by executing the following pip command:

```bash
pip install nltk
```

Subsequently, downloading NLTK data for tokenizers, stopwords, and named entity recognition is crucial. The following Python code accomplishes this:

```bash
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")
```

# Usage

## Tokenization

Tokenization involves breaking a text into individual sentences or words.

1. Tokenizing Sentences:
Sentence tokenization divides text into sentences, crucial for numerous natural language processing tasks.

```bash
from nltk.tokenize import sent_tokenize

def tokenize_sentences(text):
    return sent_tokenize(text)
```

2. Tokenizing Words:
Tokenizing words splits a sentence or text into individual words or tokens.

```bash
from nltk.tokenize import word_tokenize

def tokenize_words(text):
    return word_tokenize(text)
```

## Lowercasing

Lowercasing transforms all text to lowercase, ensuring uniformity and consistency.

```bash
def lowercase(text):
    return text.lower()
```

## Removing Stopwords

Stopwords, common words like "the" and "is," are often removed to reduce dimensionality and eliminate noise.

```bash
from nltk.corpus import stopwords

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_words = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_words)
```

## Stemming

Stemming transforms a word to its root form, reducing variations.

```bash
from nltk.stem import PorterStemmer

def stem_words(text):
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    return [(word, stemmer.stem(word)) for word in words]
```

## Lemmatization

Lemmatization reduces words to their base form, considering linguistic context.

```bash
from nltk.stem import WordNetLemmatizer

def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    token_lemma_list = [(token, lemmatizer.lemmatize(token, get_wordnet_pos(token))) for token in tokens]
    return token_lemma_list
```

## Named Entity Recognition

Named Entity Recognition (NER) identifies and categorizes named entities in text.

```bash
from nltk import ne_chunk

def named_entity_recognizer(text):
    words = word_tokenize(text)
    tagged = nltk.pos_tag(words)
    named_entities = ne_chunk(tagged)
    return named_entities
```