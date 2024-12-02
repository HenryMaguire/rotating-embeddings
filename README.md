# Surprisingly effective text classification using only embeddings and linear algebra

Embeddings are super cheap and easily available. They are extremely effective at encoding rich information into dense vectors of a fixed size. This allows semantic similarity to be captured in the geometry of the vector space, meaning that two vectors are close together if their meanings are similar, for example the vectors for "cat" and "dog" are close together. Similarly, an embedding for an image of a falcon and an image of an owl should be close together.

One naive approach to classification is to use the cosine similarity between the embeddings of the input and the embeddings of each class, in a zero-shot fashion. This is great, because it is simple and effectively parameter-free. However, if we do have some labeled data, we can do better.

In this notebook, we will explore a linear, supervised approach that just uses the singular value decomposition (SVD) to find an optimal rotation matrix that best aligns input embeddings with class embeddings. This approach can lead to dramatically better classification results than the zero-shot approach, especially when the classes are not easily separable in the embedding space. On top of this, we will use the singular values to identify the most important components of the embeddings, which allows us to compress the model by over 99% while retaining 100% of the classification accuracy.

This was inspired by [this fairly old paper](https://arxiv.org/abs/1702.03859) which used a similar approach for machine translation.


## TODO

- [ ] Generate more interesting datasets to test this on.
