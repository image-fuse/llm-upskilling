# Vector Databases

Vector databases are specialized databases designed to efficiently store, search, and retrieve high-dimensional vectors. A vector refers to a mathematical representation of data points in a multi-dimensional space. Vector embeddings are representations of data (such as words, images, or other types of information) in a continuous vector space. For example, in natural language processing, words are often represented as dense vectors where similar words are close in the vector space. These embeddings are used to capture semantic relationships between data points.

With a vector database, we can add advanced features to our AIs, like semantic information retrieval, long-term memory, and more. The diagram below gives us a better understanding of the role of vector databases in this type of application:

![https://cdn.sanity.io/images/vr8gru94/production/e88ebbacb848b09e477d11eedf4209d10ea4ac0a-1399x537.png](https://cdn.sanity.io/images/vr8gru94/production/e88ebbacb848b09e477d11eedf4209d10ea4ac0a-1399x537.png)

1. First, we use the **embedding model** to create **vector embeddings** for the **content** we want to index.
2. The **vector embedding** is inserted into the **vector database**, with some reference to the original **content** the embedding was created from.
3. When the **application** issues a query, we use the same **embedding model** to create embeddings for the query, and use those embeddings to query the **database** for *similar* vector embeddings. And as mentioned before, those similar embeddings are associated with the original **content** that was used to create them.

## **Vector Database vs. Traditional Database**:

A vector database is tailored for efficiently managing high-dimensional vectors, making it an ideal choice for applications that heavily rely on similarity searches and nearest-neighbor queries. In contrast, traditional databases, such as relational databases like MySQL and PostgreSQL, or NoSQL databases are designed for general-purpose data storage and retrieval. They are typically optimized for structured data organized in tables with rows and columns. However, they lack the specialized features needed to efficiently manage and query high-dimensional vectors, which can lead to performance challenges when dealing with tasks like similarity searches.

## ****Vector index vs Vector database?****

1. **Data management:** Vector databases offer well-known and easy-to-use features for data storage, like inserting, deleting, and updating data. This makes managing and maintaining vector data easier than using a standalone vector *index* like FAISS, which requires additional work to integrate with a storage solution.
2. **Metadata storage and filtering:** Vector databases can store metadata associated with each vector entry. Users can then query the database using additional metadata filters for finer-grained queries.
3. **Scalability:** Vector databases are designed to scale with growing data volumes and user demands, providing better support for distributed and parallel processing. Standalone vector indices may require custom solutions to achieve similar levels of scalability.
4. **Data security and access control:** Vector databases typically offer built-in data security features and access control mechanisms to protect sensitive information, which may not be available in standalone vector index solutions.

## ****How does a vector database work?****

As we know traditional databases store strings, numbers, and other types of scalar data in rows and columns. We are usually querying for rows in the database where the value usually exactly matches our query. In vector databases, we apply a similarity metric to find a vector that is the **most similar** to our query.

A vector database uses a combination of different algorithms that all participate in Approximate Nearest Neighbor (ANN) search. These algorithms optimize the search through hashing, quantization, or graph-based search.

![common pipeline for a vector database](https://cdn.sanity.io/images/vr8gru94/production/ff9ba425d0c78d696372e0a43ce57851b4f1d4b7-1307x233.png)

common pipeline for a vector database

1. **Indexing:** This step maps the vectors to a data structure that will enable faster searching. Algorithms such as PQ, LSH, or HNSW are used for indexing vectors.
2. **Querying**: The vector database compares the indexed query vector to the indexed vectors in the dataset to find the nearest neighbors (applying a similarity metric used by that index).
3. **Post Processing**: In some cases, the vector database retrieves the final nearest neighbors from the dataset and post-processes them to return the final results. This step can include re-ranking the nearest neighbors using a different similarity measure.

## Similarity Measures in Vector database

Similarity measures are mathematical methods for determining how similar two vectors are in a vector space.

Several similarity measures can be used, including:

- **Cosine similarity:** measures the cosine of the angle between two vectors in a vector space. It ranges from -1 to 1, where 1 represents identical vectors, 0 represents orthogonal vectors, and -1 represents vectors that are diametrically opposed.
- **Euclidean distance:** measures the straight-line distance between two vectors in a vector space. It ranges from 0 to infinity, where 0 represents identical vectors, and larger values represent increasingly dissimilar vectors.
- **Dot product:** measures the product of the magnitudes of two vectors and the cosine of the angle between them. It ranges from -∞ to ∞, where a positive value represents vectors that point in the same direction, 0 represents orthogonal vectors, and a negative value represents vectors that point in opposite directions.

# Comparative Study of Vector Databases

## FAISS

Faiss is an open-source library — developed by Facebook AI — that enables efficient similarity search. Given a set of vectors, we can index them using Faiss — then using another vector (the query vector), we search for the most similar vectors within the index.

### Key Features

- **Fast similarity search**: FAISS offers highly optimized algorithms for efficient similarity search. It supports both exact and approximate nearest neighbor search, allowing users to trade off between search accuracy and computational efficiency.
- **GPU support:** FAISS leverages the computational power of GPUs to accelerate similarity search operations.
- **Extensibility**: FAISS is designed to be modular and extensible, allowing users to customize and extend its functionality based on their specific requirements. It provides a flexible API that supports different indexing methods, distance metrics, and search parameters.

## Milvus

Milvus is an open-source vector database. It supports adding, deleting, updating, and near real-time search of vectors on a trillion-byte scale.

Milvus runs on a client-server model.

- The Milvus server includes the Milvus Core and Meta Store.
    - Milvus Core stores and manages vectors and scalar data.
    - Meta Store stores and manages metadata in SQLite for testing or MySQL for production.
- On the client side, Milvus provides SDKs in Python, Java, Go, and C++, as well as RESTful APIs.

### Key Features

- Optimizes search and indexing performance on GPU.
- Searches trillion-byte scale datasets in milliseconds.
- Manages inserting, deleting, updating, and querying vector data in a dynamic environment.
- Offers support for Faiss, NMSLIB, and Annoy libraries.
- Measures similarity using Euclidean distance (L2), inner product, Hamming distance, Jaccard distance, and more.
- **Near-real-time (NRT) search:** Newly inserted datasets are available for search in one second or less.

## Pinecone

Pinecone is a managed vector database designed to handle real-time search and similarity matching at scale. 

**Pros**

1. **Real-time search:** Pinecone offers blazing-fast search capabilities, allowing users to retrieve similar vectors in real-time, making it well-suited for applications like recommendation engines and content-based searching.
2. **Scalability:** Pinecone’s architecture is built to scale with growing data and traffic demands, making it an excellent choice for high-throughput applications that deal with vast amounts of data.
3. **Automatic indexing:** Pinecone automatically indexes vectors, reducing the burden on developers and simplifying the deployment process.
4. **Python support:** Pinecone provides an easy-to-use Python SDK, making it accessible to developers and data scientists familiar with the Python ecosystem.

**Cons**

1. **Cost:** As a managed service, Pinecone’s pricing structure might be a concern for some users, particularly for large-scale deployments with significant data volumes.
2. **Limited querying functionality:** While Pinecone excels at similarity search, it might lack some advanced querying capabilities that certain projects require.

## ChromaDB

Chroma is the open-source vector database. It offers a robust set of features that cater to various use cases, making it a viable choice for many vector-based applications.

Chroma allows more flexible querying capabilities, including complex range searches and combinations of vector attributes, making it suitable for a broader range of applications.

**Cons**

1. **Deployment complexity:** Setting up Chroma and managing it at scale might require more effort and expertise compared to a managed solution like Pinecone.
2. **Performance considerations:** While Chroma is efficient for many use cases, it might not match Pinecone’s performance in certain high-throughput real-time scenarios.

# References

- [What is Vector Database?](https://www.pinecone.io/learn/vector-database/)
- [Vector Database Comparison Cheatsheet](https://docs.google.com/spreadsheets/d/1oAeF4Q7ILxxfInGJ8vTsBck3-2U9VV8idDf3hJOozNw/edit?usp=sharing)
- [FAISS](https://medium.com/@aravindariharan/faiss-ai-similarity-search-6a70d6f8930b)
- [What is Milvus](https://milvus.io/docs/v1.1.1/overview.md)
- [Pinecone vs Chroma](https://medium.com/@woyera/pinecone-vs-chroma-the-pros-and-cons-2b0b7628f48f)