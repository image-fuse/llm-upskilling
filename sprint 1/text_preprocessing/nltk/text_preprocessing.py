import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import ne_chunk
from nltk.corpus import wordnet

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")


def tokenize_sentences(text):
    """
    Tokenizes input text into sentences.

    Args:
        text (str): The input text.

    Returns:
        list: A list of tokenized sentences.
    """
    return sent_tokenize(text)


def tokenize_words(text):
    """
    Tokenizes input text into words.

    Args:
        text (str): The input text.

    Returns:
        list: A list of tokenized words.
    """
    return word_tokenize(text)


def lowercase(text):
    """
    Converts input text to lowercase.

    Args:
        text (str): The input text.

    Returns:
        str: The input text in lowercase.
    """
    return text.lower()


def remove_stopwords(words):
    """
    Removes stopwords from a list of words.

    Args:
        words (list): A list of words.

    Returns:
        list: A list of words with stopwords removed.
    """
    stop_words = set(stopwords.words("english"))
    # tokenize
    word_tokens = word_tokenize(text)
    filtered_words = [
        word for word in word_tokens if word not in stop_words]

    return ' '.join(filtered_words)


def stem_words(words):
    """
    Stems words in a list.

    Args:
        words (list): A list of words.

    Returns:
        list: A list of stemmed words.
    """
    stemmer = PorterStemmer()
    # tokenize
    words = word_tokenize(text)
    return [(word, stemmer.stem(word))for word in words]

# Lemmatization


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_words(text):
    """
    Lemmatizes words in the input text.

    Args:
        text (str): The input text.

    Returns:
        list of tuple: A list containing the original word and lemma.
    """

    # 1. Init Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # tokenize words
    tokens = word_tokenize(text)

    # 1. Init Lemmatizer
    lemmatizer = WordNetLemmatizer()

    token_lemma_list = [(token, lemmatizer.lemmatize(
        token, get_wordnet_pos(token))) for token in tokens]

    return token_lemma_list


def named_entity_recognizer(text):
    """
    Extracts named entities from input text.

    Args:
        text (str): The input text.

    Returns:
        nltk.Tree: A tree structure representing named entities.
    """
    words = word_tokenize(text)
    tagged = nltk.pos_tag(words)
    named_entities = ne_chunk(tagged)
    return named_entities


if __name__ == "__main__":
    text = """
    The United States of America, commonly known as the United States or
    America, is a country primarily located in North America and
    consisting of 50 states, a federal district, five major
    unincorporated territories, nine Minor Outlying Islands,
    and 326 Indian reservations. It is the world's
    third-largest country by both land and total area.
    """
    # Tokenize sentences
    sentences = tokenize_sentences(text)
    print("Sentences: ")
    print(sentences)

    # Tokenize words
    words = tokenize_words(text)
    print("\nWords:")
    print(words)

    # Lowercase
    lowercased_text = lowercase(text)
    print("\nLowercased Text:")
    print(lowercased_text)

    # Remove stopwords
    filtered_words = remove_stopwords(text)
    print("\nWords after Stopword Removal:")
    print(filtered_words)

    # Stemming
    stemmed_words = stem_words(text)
    print("\nStemmed Words:")
    print(stemmed_words)

    # Lemmatization
    lemmatized_words = lemmatize_words(text)
    print("\nLemmatized Words: ")
    print(lemmatized_words)

    # Named Entity Recognition
    named_entities = named_entity_recognizer(text)
    print("\nNamed Entities:")
    print(named_entities)
