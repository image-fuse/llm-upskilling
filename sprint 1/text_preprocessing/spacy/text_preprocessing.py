import spacy

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")


def tokenize_sentences(text):
    """
    Tokenizes input text into sentences.

    Args:
        text (str): The input text.

    Returns:
        list: A list of tokenized sentences.
    """
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


def tokenize_words(text):
    """
    Tokenizes input text into words.

    Args:
        text (str): The input text.

    Returns:
        list: A list of tokenized words.
    """
    doc = nlp(text)
    return [token.text for token in doc]


def lowercase(text):
    """
    Converts input text to lowercase.

    Args:
        text (str): The input text.

    Returns:
        str: The input text in lowercase.
    """

    text_to_lower = text.lower()

    return text_to_lower


def remove_stopwords(text):
    """
    Removes stopwords from input text.

    Args:
        text (str): The input text.

    Returns:
        str: Input text with stopwords removed.
    """
    doc = nlp(text)
    filtered_text = " ".join(token.text for token in doc if not token.is_stop)
    return filtered_text


# Lemmatization


def lemmatize_word(text):
    """
    Lemmatizes words in the input text.

    Args:
        text (str): The input text.

    Returns:
        list: A list of lemmatized words.
    """
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc]
    return lemmatized_words


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


if __name__ == "__main__":
    text = """
    The United States of America, commonly known as the United States or
    America, is a country primarily located in North America and
    consisting of 50 states, a federal district, five major
    unincorporated territories, nine Minor Outlying Islands,
    and 326 Indian reservations. It is the world's
    third-largest country by both land and total area.
    """

    # Tokenize sentences using spaCy
    sentences = tokenize_sentences(text)
    print("Sentences:")
    print(sentences)

    # Tokenize words using spaCy
    words = tokenize_words(text)
    print("\nWords:")
    print(words)

    # Lowercase using spaCy
    lowercased_text = lowercase(text)
    print("\nLowercased Text:")
    print(lowercased_text)

    # Remove stopwords using spaCy
    filtered_text = remove_stopwords(text)
    print("\nText after Stopword Removal:")
    print(filtered_text)

    # Lemmatization using spaCy
    lemmatized_words = lemmatize_word(text)
    print("\nLemmatized Words:")
    print(lemmatized_words)

    # Named Entity Recognition using spaCy
    named_entities = named_entity_recognition(text)
    print("\nNamed Entities:")
    for entity in named_entities:
        print(f"Text: {entity.text}, Label: {entity.label_}")
