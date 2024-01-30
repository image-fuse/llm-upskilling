# Word Embeddings

Word embeddings are numerical representations of words. These representations are designed to capture the semantic and syntactic relationships between words, making it easier for computers to understand and work with words in natural language processing (NLP) tasks. As machines don’t understand words or sentences and can only process numbers, we must encode text data as numbers for input or output for any machine.

# Frequency Based Embedding


## One-Hot encoding

Every word (even symbols) which are part of the given text data are written in the form of vectors, constituting only of 1 and 0. This allows the word to be identified uniquely by its one hot vector and vice versa, that is no two words will have same one hot vector representation.

![Untitled](https://editor.analyticsvidhya.com/uploads/87029one_hot_encoding_demo.png)

## Bag of Words (BOW)

It represents a text document as an unordered collection of words or tokens, disregarding grammar, word order, and context. The primary idea behind the Bag of Words model is to count the frequency of each word in a document and represent the document as a vector of word counts. Now there are 2 kinds of BOW:

1. Binary BOW
2. BOW

The difference between them is, in Binary BOW we encode 1 or 0 for each word appearing or non-appearing in the sentence. We do not take into consideration the frequency of the word appearing in that sentence.

![https://miro.medium.com/v2/resize:fit:720/format:webp/1*9n9GEWBTKRwDAYbj8GwDHw.png](https://miro.medium.com/v2/resize:fit:720/format:webp/1*9n9GEWBTKRwDAYbj8GwDHw.png)

## TF-IDF

TF-IDF is a statistical measure used to evaluate the importance of a word within a document relative to a collection of documents (corpus). It considers both the term frequency (how often a word appears in a document) and the inverse document frequency (how common or rare a word is across the entire corpus).

************************Term Frequency:************************ the occurrence of the current word in the current sentence w.r.t the total number of words in the current sentence.

![https://miro.medium.com/v2/resize:fit:438/format:webp/1*6i6qxHy2KpiSwyn7_kx7Gw.png](https://miro.medium.com/v2/resize:fit:438/format:webp/1*6i6qxHy2KpiSwyn7_kx7Gw.png)

**Inverse Document Frequency**: Log of Total number of words in the whole data corpus w.r.t the total number of sentences containing the current word.

![https://miro.medium.com/v2/resize:fit:604/format:webp/1*eb2S75KY9avM-gxvICuhEg.png](https://miro.medium.com/v2/resize:fit:604/format:webp/1*eb2S75KY9avM-gxvICuhEg.png)

Suppose we have a corpus containing three documents:

Document 1: "The quick brown fox jumps over the lazy dog."

Document 2: "A brown dog barks loudly."

Document 3: "The dog chases the fox."

We'll perform TF-IDF feature extraction for the word "brown".

****Calculate Term Frequency (TF)****

For the word "brown" in each document:

TF("brown", Document 1) = 1/9

TF("brown", Document 2) = 1/4

TF("brown", Document 3) = 0

****Calculate Inverse Document Frequency (IDF):****

IDF("brown") = $log(3/2) \approx 0.176$

 ****Calculate TF-IDF****

TF-IDF("brown", Document 1) = $1/9 \times 0.176 \approx 0.0196$

TF-IDF("brown", Document 2) = $1/4 \times 0.176 \approx 0.044$

TF-IDF("brown", Document 3) = $0 \times 0.176 \approx 0$

# Prediction Based Embedding

## Word2Vec

Word2vec is a technique for natural language processing (NLP) published in 2013. The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words. They take large corpus of text as input and produces a vector space with each unique word in the corpus being assigned a corresponding vector in the space.

Word2vec can utilize either of two model architectures to produce embeddings: CBOW(Continuous bag of words) and Skip-gram. In both architectures, word2vec considers both individual words and a sliding context window as it iterates over the corpus.

![https://miro.medium.com/v2/resize:fit:720/format:webp/1*fxu9SO1rx3vQjbBnDYFvjA.png](https://miro.medium.com/v2/resize:fit:720/format:webp/1*fxu9SO1rx3vQjbBnDYFvjA.png)

### **CBOW (Continuous Bag of words)**

The way CBOW work is that it tends to predict the probability of a target word based on its surrounding context words.

**Architecture**

1. **Input Layer**: Receives one-hot encoded vectors representing context words.
2. **Hidden Layer**: A single hidden layer that contains the word embeddings (vector representations).
3. **Output Layer**: Predicts the target word using a softmax activation function.

### ****Skip-gram****

Skip-gram is another word2vec model that aims to predict the surrounding context words given a target word. It reverses the process of CBOW by using a single target word to predict multiple context words.

### **Architecture**

1. **Input Layer**: Receives a one-hot encoded vector representing the target word.
2. **Hidden Layer**: Contains the word embeddings (vector representations).
3. **Output Layer**: Predicts context words using a softmax activation function.

**CBOW vs. Skip-gram**:

- CBOW is faster to train, but skip-gram performs better on rare words.
- CBOW works well with small datasets and frequent words, while skip-gram is more suitable for larger datasets.
- CBOW is considered to be better for tasks like part-of-speech tagging and named entity recognition, while skip-gram is better for word similarity tasks.

## Glove

GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. 

GloVe starts with constructing a co-occurrence matrix from a large corpus of text. This matrix represents how often words co-occur in the same context (e.g., within a certain window of words).

GloVe aims to learn word embeddings that preserve the ratio of co-occurrence probabilities. It formulates an objective function that minimizes the difference between the dot product of the word vectors and the logarithm of the co-occurrence probabilities.

### **Advantages of GloVe**

1. **Global Context**: GloVe considers the entire corpus during training, capturing global word relationships. This is especially useful for tasks where understanding word relationships beyond local context is important.
2. **Effective for Common and Rare Words**: GloVe tends to handle both common and rare words well due to its use of co-occurrence counts.

## ELMO (Embeddings from Language Models)

Elmo (Embeddings from Language Models) is a contextualized word embedding technique introduced by the Allen Institute for Artificial Intelligence in 2018. Unlike traditional word embeddings that assign a fixed vector to each word, Elmo provides embeddings that vary based on the context in which the word appears. Elmo is based on a bidirectional Long Short-Term Memory (LSTM) network. This architecture enables it to consider both left and right contexts when generating embeddings.

ELMo word vectors are computed on top of a two-layer bidirectional language model (biLM). This biLM model has two layers stacked together. Each layer has 2 passes — forward pass and backward pass:

![https://cdn.analyticsvidhya.com/wp-content/uploads/2019/03/output_YyJc8E.gif](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/03/output_YyJc8E.gif)

- The architecture above uses a character-level convolutional neural network (CNN) to represent words of a text string into raw word vectors
- These raw word vectors act as inputs to the first layer of biLM
- The forward pass contains information about a certain word and the context (other words) before that word
- The backward pass contains information about the word and the context after it
- This pair of information, from the forward and backward pass, forms the intermediate word vectors
- These intermediate word vectors are fed into the next layer of biLM
- The final representation (ELMo) is the weighted sum of the raw word vectors and the 2 intermediate word vectors

## BERT

BERT’s model architecture is based on Transformers. It uses multilayer bidirectional transformer encoders for language representations. BERT generates contextualized word embeddings. It considers the entire sentence or context in which a word appears to generate a representation for that word. This means that the same word can have different embeddings depending on the context in which it's used.

### Input Formatting

1. A **special token**, [SEP], to mark the end of a sentence, or the separation between two sentences
2. A **special token**, [CLS], at the beginning of our text. This token is used for classification tasks, but BERT expects it no matter what your application is.
3. Tokens that conform with the fixed vocabulary used in BERT
4. The **Token IDs** for the tokens, from BERT’s tokenizer
5. **Mask IDs** to indicate which elements in the sequence are tokens and which are padding elements
6. **Segment IDs** used to distinguish different sentences
7. **Positional Embeddings** used to show token position within the sequence

### Tokenization

The BERT tokenizer was created with a WordPiece model. To tokenize a word under this model, the tokenizer first checks if the whole word is in the vocabulary. If not, it tries to break the word into the largest possible subwords contained in the vocabulary, and as a last resort will decompose the word into individual characters.

So, rather than assigning “embeddings” and every other out of vocabulary word to an overloaded unknown vocabulary token, we split it into subword tokens [‘em’, ‘##bed’, ‘##ding’, ‘##s’] that will retain some of the contextual meaning of the original word. We can even average these subword embedding vectors to generate an approximate vector for the original word.

### Extracting Embeddings

We can use the pre-trained BERT to create contextualized word embeddings. Then we can feed these embeddings to our existing model

![https://jalammar.github.io/images/bert-contexualized-embeddings.png](https://jalammar.github.io/images/bert-contexualized-embeddings.png)

# Sentence Transformers

Sentence Transformers are a type of models designed to generate high-quality embeddings for entire sentences or longer text passages. Unlike traditional word embeddings, which represent individual words, sentence embeddings capture the semantic meaning of entire sentences. This is valuable for various natural language processing tasks, such as semantic similarity, text classification, and information retrieval.

These models use techniques similar to BERT but are adapted to focus on sentence-level tasks. They are trained on large datasets containing pairs of sentences, learning to produce embeddings that bring similar sentences closer together in the embedding space.

With BERT to calculate accurate sentence similarity, the approach was to use cross-encoder structure. This meant that we would pass two sentences to BERT, add a classification head to the top of BERT — and use this to output a similarity score.

![https://cdn.sanity.io/images/vr8gru94/production/9a89f1b7dddd4c78da8b9ba0311c2ffd1ff18ffe-1920x1080.png](https://cdn.sanity.io/images/vr8gru94/production/9a89f1b7dddd4c78da8b9ba0311c2ffd1ff18ffe-1920x1080.png)

The BERT cross-encoder architecture consists of a BERT model which consumes sentences A and B. Both are processed in the same sequence, separated by a [SEP] token. All of this is followed by a feedforward NN classifier that outputs a similarity score.

The cross-encoder network is very accurate by not scalable.

The solution to this lack of an accurate model with reasonable latency was designed by Nils Reimers and Iryna Gurevych in 2019 with the introduction of sentence-BERT (SBERT) and the sentence-transformers library.

SBERT is fine-tuned on sentence pairs using a siamese architecture.
uses mean pooling on the final output layer to produce a sentence embedding.

![https://cdn.sanity.io/images/vr8gru94/production/2425dc0efd3f73a0bf57b3bf85a091c78619ec2c-1920x1110.png](https://cdn.sanity.io/images/vr8gru94/production/2425dc0efd3f73a0bf57b3bf85a091c78619ec2c-1920x1110.png)

**Siamese BERT Pre-Training**

---

The softmax-loss approach used the ‘siamese’ architecture fine-tuned on the Stanford Natural Language Inference (SNLI) and Multi-Genre NLI (MNLI) corpora.

SNLI contains 570K sentence pairs, and MNLI contains 430K. The pairs in both corpora include a premise and a hypothesis. Each pair is assigned one of three labels:

- 0 — entailment, e.g. the premise suggests the hypothesis.
- 1 — neutral, the premise and hypothesis could both be true, but they are not necessarily related.
- 2 — contradiction, the premise and hypothesis contradict each other.

Given this data, we feed sentence A (let’s say the premise) into siamese BERT A and sentence B (hypothesis) into siamese BERT B.

The siamese BERT outputs our pooled sentence embeddings. The mean-pooling approach was best performing for both NLI and STSb datasets.

There are now two sentence embeddings. We will call embeddings A `u` and embeddings B `v`. The next step is to concatenate u and v. Again, several concatenation approaches were tested, but the highest performing was a `(u, v, |u-v|)` operation:

|u-v| is calculated to give us the element-wise difference between the two vectors. The `u`, `v` and `|u-v|` vectors all fed into feedforward neural network that has three outputs which aligh with NLI similarity labels 0, 1, and 2.

Then, we need to calculate the softmax from our FFNN which is done within the cross-entropy loss function. The softmax and labels are used to optimize on this ‘softmax-loss’.

![https://cdn.sanity.io/images/vr8gru94/production/a7bc429139dfb58998cee4fe84341ef5b66f2019-1920x990.png](https://cdn.sanity.io/images/vr8gru94/production/a7bc429139dfb58998cee4fe84341ef5b66f2019-1920x990.png)

# References

- [Analytics Vidhya - Word Embeddings](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/)
- [Text-Encoding Blog](https://medium.com/analytics-vidhya/nlp-text-encoding-a-beginners-guide-fa332d715854)
- [Word2Vec Scratch Implementation](https://aegis4048.github.io/demystifying_neural_network_in_skip_gram_language_modeling)
- [Hands on Guide to word embeddings using glove](https://analyticsindiamag.com/hands-on-guide-to-word-embeddings-using-glove/)
- [ELMO Embeddings Blog](https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/)
- [Hands on BERT Word Embeddings](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/)
- [Sentence Embeddings Pinecone Documentation](https://www.pinecone.io/learn/series/nlp/sentence-embeddings/)
- [sbert Documentation](https://www.sbert.net/)
- [Illustrated Bert](https://jalammar.github.io/illustrated-bert/)