# Transformer and BERT

# Transformers

### Introduction

The Transformer is a model leveraging attention to enhance the training speed and outperforming traditional models like the Google Neural Machine Translation model. It stands out for its parallelization capabilities.

### High-Level Architecture

The Transformer is represented as a black box with encoding and decoding components connected by attention layers. Both components consist of stacked identical layers – encoders for the input and decoders for the output.
![high-level-architecture](https://jalammar.github.io/images/t/the_transformer_3.png)
In more detail, we can see the transformer has an encoding component, a decoding component and connections between them.
![](https://jalammar.github.io/images/t/The_transformer_encoders_decoders.png)  

**Encoders**

All encoders are identical in structure though they don’t share the weights. Each of the encoders is broken down into two sub layers:

- an self attention layer
- and a feed forward neural network

The input to the encoder first goes through a self attention layer.

- attention layer helps the encoder to look at other words in the input sentence as it encodes a specific word.

The outputs of the self-attention layer are fed to a feed-forward neural network.

![https://jalammar.github.io/images/t/Transformer_decoder.png](https://jalammar.github.io/images/t/Transformer_decoder.png)

Encoder receives a list of vectors as input. Encoder processes this list of vectors by passing them

- into a self-attention layer
- then into a feed-forward neural network
- then sends out the output upwards to the next encoder.

![https://jalammar.github.io/images/t/encoder_with_tensors_2.png](https://jalammar.github.io/images/t/encoder_with_tensors_2.png)

**Decoders**

The decoder has both the layers of encoders and between them another attention layer that helps the decoder focus on relevant parts of the input sentence.

# Embeddings

Like other NLP applications, we begin by turning each input word into a vector using an [embedding algorithm](https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca). The embedding happens only in the the bottom most encoder.

The encoders receive a list of vectors each of size 512.

- The bottom encoder receives word embeddings.
- Other encoders receives the output of the encoder that’s directly below.

After embedding the words in our input sequence, each of them flows through each of the two layers of the encoder.

![https://jalammar.github.io/images/t/encoder_with_tensors.png](https://jalammar.github.io/images/t/encoder_with_tensors.png)

The word in each position flows through its own path in the encoder. There are dependencies between these paths in the self-attention layer but not in feed-forward layer.

Thus, various paths can be executed in parallel while flowing through the feed-forward layer.

### Self-Attention

The encoder's self-attention layer allows the model to consider other words in the input sentence when encoding a specific word. Self-attention aids in capturing dependencies between words, unlike feed-forward layers that operate independently.

Let’s take an input sentence for translation:

”`The animal didn't cross the street because it was too tired`”

When the model is processing the word “it”, self-attention allows it to associate “it” with “animal”.

### How to calculate:

**Query, Key, Value Vectors**

- Create three vectors from each of the encoder’s input vectors. We call them, Query vector, Key vector, and Value Vector.
- They are created by multiplying the embeddings by three matrices that we train during training process.
- The new vectors are smaller in dimension than the embedding vectors.

![https://jalammar.github.io/images/t/transformer_self_attention_vectors.png](https://jalammar.github.io/images/t/transformer_self_attention_vectors.png)

**Score**

- If we are calculating for the word ‘thinking’ we need to score each word against this word.
- The score determines how much focus to place on other parts of input sentence as we encode a word at a certain position.
  - Calculate dot product of the query vector with the key vector of the respective word we’re scoring.
    - if we’re processing the self-attention for the word in position #1, the first score would be the dot product of q1 and k1. The second score would be the dot product of q1 and k2.
  ![https://jalammar.github.io/images/t/transformer_self_attention_score.png](https://jalammar.github.io/images/t/transformer_self_attention_score.png)
- Now divide the scores by the square root of the dimension of the key vectors (8).
  - For stable gradients.
- Then, pass the result through a softmax operation.
  - Softmax normalizes the scores so they’re all positive and add up to 1.
  ![https://jalammar.github.io/images/t/self-attention_softmax.png](https://jalammar.github.io/images/t/self-attention_softmax.png)
- Multiply each value vector by the softmax score.
  - to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words
- Sum up the weighted value vectors. This produces the output of the self-attention layer at this position (for the first word).
  ![https://jalammar.github.io/images/t/self-attention-output.png](https://jalammar.github.io/images/t/self-attention-output.png)

![https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png](https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png)

## Multi-Headed Attention.

With multi-headed attention, we have multiple sets of Query/Key/Value weight matrices. Each of these sets is randomly initialized. Then, after training, each set is used to project the input embeddings into a different representation subspace.

This improves the performance of the attention layer in two ways:

1. It expands the model’s ability to focus on different positions.
2. It gives the attention layer multiple “representation subspaces”.

![With multi-headed attention, we maintain separate Q/K/V weight matrices for each head resulting in different Q/K/V matrices](https://jalammar.github.io/images/t/transformer_attention_heads_qkv.png)

With multi-headed attention, we maintain separate Q/K/V weight matrices for each head resulting in different Q/K/V matrices

Doing the above outlined calculation for self-attention, we would end up with eight different z matrices. But the feed-forward layer is not expecting eight matrices.

So, we concat the matrices then multiply them by an additional weights matrix WO.

![https://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png](https://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png)

![https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png](https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)

### Positional Encoding

Positional encoding addresses the model's inability to understand the order of words in a sequence. This involves adding a vector to each input embedding, providing information about the position of each word.

These vectors follow a specific pattern that the model learns, which helps it determine the position of each word, or the distance between different words in the sequence.

these values provides meaningful distances between the embedding vectors once they’re projected into Q/K/V vectors and during dot-product attention.

![https://jalammar.github.io/images/t/transformer_positional_encoding_example.png](https://jalammar.github.io/images/t/transformer_positional_encoding_example.png)

### Residuals and normalization layer

Residual connections and layer normalization are introduced as essential components in each sub-layer (self-attention, feed-forward) of both the encoder and decoder. They contribute to stable training.

![https://jalammar.github.io/images/t/transformer_resideual_layer_norm.png](https://jalammar.github.io/images/t/transformer_resideual_layer_norm.png)

If we’re to think of a Transformer of 2 stacked encoders and decoders, it would look something like this:

![https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png)

### Decoder

The decoding phase involves transforming the encoder's output into attention vectors for the decoder. Self-attention in the decoder is restricted to earlier positions in the output sequence.

Let us see how the encoder and decoder work together. The encoder start by processing the input sequence. The output of the top encoder is then transformed into a set of attention vectors K and V. These are used by each decoder in its “encoder-decoder attention” layer which helps the decoder focus on appropriate places in the input sequence.

![https://jalammar.github.io/images/t/transformer_decoding_1.gif](https://jalammar.github.io/images/t/transformer_decoding_1.gif)

The following steps repeat the process until a special symbol is reached which indicates the transformer has completed its output.

- The output of each step is fed to the bottom decoder in the next time step
- We also embed and add positional encoding to those decoder inputs to indicate the position of each word.

![https://jalammar.github.io/images/t/transformer_decoding_2.gif](https://jalammar.github.io/images/t/transformer_decoding_2.gif)

In the decoder, the self-attention layer is only allowed to attend to earlier positions in the output sequence.

- This is done by masking future positions before the softmax step in the self-attention calculation.

The “Encoder-Decoder Attention” layer works just like multiheaded self-attention, except

- it creates its Queries matrix from the layer below it, and takes the Keys and Values matrix from the output of the encoder stack.

### Final Linear and Softmax Layer

The decoder stack outputs a vector of floats, projected into a larger logits vector by the final linear layer. The softmax layer converts these scores into probabilities for word generation.

### Training

Training involves comparing the model's output probability distributions with the desired output. The loss function measures the dissimilarity, and backpropagation adjusts the model's weights to minimize this dissimilarity.

### Loss Function

The loss function involves comparing the model's probability distributions with the expected output distributions. Cross-entropy and Kullback–Leibler divergence are mentioned as relevant metrics.

# BERT

**BERT** (Bidirectional Encoder Representation From Transformer) is a transformers model pre-trained on a large corpus of English data in a self-supervised fashion. 

BERT makes use of Transformer, an attention mechanism that learns contextual relations between words (or sub-words) in a text. Transformer includes two separate mechanisms — an encoder that reads the text input and a decoder that produces a prediction for the task.

BERT only uses the encoder as its goal is to generate a language model.

**BERT works in two steps:**

#### Pretraining:
- **Architecture**: BERT is built upon a deep neural network architecture known as the Transformer.
- **Massive Corpus**: BERT is pre-trained on a massive corpus of text, often containing billions of words, to learn the fundamental properties of language. The model is trained to predict the probability of a word or token in a sentence based on the surrounding words, taking into account both the words to the left and right of it. This bidirectional context modeling is a key feature of BERT.
- **Masked Language Modeling (MLM)**: During pretraining, BERT learns by predicting masked words in sentences. Some words in the input text are randomly replaced with a special [MASK] token, and the model's objective is to predict what the masked words are based on the surrounding context. This process helps BERT capture a deep understanding of word meanings and context.
- **Segment Embeddings**: BERT also incorporates segment embeddings to distinguish between different sentences or segments of text within a document. This is particularly useful for tasks that involve multiple sentences, such as question-answering.
- **Positional Encodings**: Transformers, including BERT, do not have built-in knowledge of word order. Positional encodings are added to the input embeddings to provide information about the position of each word in the sequence.
- **Layer Stacking**: BERT consists of multiple layers of transformers stacked on top of each other. Each layer refines the representation of the input text.

#### Fine-Tuning:
- After the pretraining phase, BERT is a language model with a deep architecture and a strong understanding of language. However, it doesn't know specifics about any particular NLP task.
- To make BERT useful for specific tasks like text classification, question answering, or sentiment analysis, it undergoes a fine-tuning process.
- During fine-tuning, BERT's pre-trained weights are further trained on a smaller dataset related to the specific task at hand. This dataset contains labeled examples for the task.
- The fine-tuning process adjusts the model's weights to make it perform well on the target task. The model learns task-specific patterns and features from the fine-tuning data.
- Fine-tuning typically involves adding a task-specific output layer to the pre-trained BERT model, which transforms BERT's contextual embeddings into task-specific predictions. For example, in text classification, the output layer might consist of a softmax layer for classifying text into different categories.

### Model Architecture

The paper presents two model sizes for BERT:

- BERT BASE – Comparable in size to the OpenAI Transformer in order to compare performance
- BERT LARGE – A ridiculously huge model which achieved the state of the art results reported in the paper

![https://miro.medium.com/v2/resize:fit:720/format:webp/0*ws9kRab0kLculhYx.png](https://miro.medium.com/v2/resize:fit:720/format:webp/0*ws9kRab0kLculhYx.png)

BERT is a Transformer encoder stack. Both BERT-base and BERT-large models are pre-trained on large text corpora and can be fine-tuned for specific downstream NLP tasks, such as text classification, named entity recognition and question answering. The larger BERT-large model generally provides improved performance over the BERT-base due to its increased model capacity, but it also requires more computational resources for training and inference.

## Model Inputs

Because BERT is a pretrained model that expects input data in a specific format, we will need:

1. A **special token, `[SEP]`,** to mark the end of a sentence, or the separation between two sentences
2. A **special token, `[CLS]`,** at the beginning of our text. This token is used for classification tasks, but BERT expects it no matter what your application is.
3. Tokens that conform with the fixed vocabulary used in BERT
4. The **Token IDs** for the tokens, from BERT’s tokenizer
5. **Mask IDs** to indicate which elements in the sequence are tokens and which are padding elements
6. **Segment IDs** used to distinguish different sentences
7. **Positional Embeddings** used to show token position within the sequence

### ****Special Tokens****

- BERT can take as input either one or two sentences, and uses the special token `[SEP]` to differentiate them.
- The `[CLS]` token always appears at the start of the text, and is specific to classification tasks.

Both tokens are *always required,* even if we only have one sentence, even if we are not using BERT for classification.

****************Example:****************

`[CLS] The man went to the store. [SEP] He bought a gallon of milk.`

### ****Tokenization****

BERT uses WordPiece Tokenization. This model greedily creates a fixed-size vocabulary of individual characters, subwords, and words that best fits our language data. The generated vocabulary contains:

1. Whole words
2. Subwords occuring at the front of a word or in isolation (“em” as in “embeddings” is assigned the same vector as the standalone sequence of characters “em” as in “go get em” )
3. Subwords not at the front of a word, which are preceded by ‘##’ to denote this case
4. Individual characters

To tokenize a word under this model:

- tokenizer first checks if the whole word is in the vocabulary.
- If not, it tries to break the word into the largest possible subwords contained in the vocabulary, and as a last resort will decompose the word into individual characters.

because of this, we can always represent a word as, at the very least, the collection of its individual characters. As a result, rather than assigning out of vocabulary words to a catch-all token like ‘OOV’ or ‘UNK,’ words that are not in the vocabulary are decomposed into subword and character tokens that we can then generate embeddings for.

### **Formalizing Input**

Each input token to the BERT is the sum of Token embeddings, Segment embeddings, and Position embeddings.

- **Position Embeddings:** Similar to the transformer, we will feed all the word sequences in the input sentence at once to the BERT model. So to identify the position of words in an input sentence i.e. where each word is located in the input sentence, we will generate position embeddings.
- **Segment Embeddings:** Segment embeddings help to indicate whether a sentence is first or second. Segment Embeddings is important because we also accomplish the `Next Sentence Prediction` task in BERT pretraining phase. So, if you want to process two sentences, assign each word in the first sentence plus the `[SEP]` token a series of 0’s, and all tokens of the second sentence a series of 1’s.
- **Token Embeddings:** Initial low-level embedding of tokens. Initial low-level embedding is essential because the machine learning model doesn’t understand textual data.

# ****BERT: Pretraining****

### ****Masked Language Modeling (MLM):****

In masked language modeling, the model randomly chooses 15% of the words in the input sentence and among those randomly chosen words, masks them 80% of the time (i.e. using **`[MASK]`** token from vocabulary), replace them with a random token 10% of the time, or keep as is 10% of the time and the model has to predict the masked words in the output.

![http://jalammar.github.io/images/BERT-language-modeling-masked-lm.png](http://jalammar.github.io/images/BERT-language-modeling-masked-lm.png)

### ****Next Sentence Prediction (NSP):****

In the Next Sentence Prediction task, Given two input sentences, the model is then trained to recognize if the second sentence follows the first one or not. helps the BERT model handle relationships between multiple sentences which is crucial for a downstream task like Q/A and ***Natural language inference***.

![http://jalammar.github.io/images/bert-next-sentence-prediction.png](http://jalammar.github.io/images/bert-next-sentence-prediction.png)

# BERT: Fine-tuning

Fine-tuning in BERT involves using pre-learned weights from the initial training phase in downstream tasks with minimal adjustments to the architecture. This allows for inexpensive training relative to the pre-training phase.

![https://miro.medium.com/v2/resize:fit:640/format:webp/0*S2YP63OvbKHI4CzT.png](https://miro.medium.com/v2/resize:fit:640/format:webp/0*S2YP63OvbKHI4CzT.png)