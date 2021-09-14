# Neural Topic Modelling
Topic modelling is usually done through statistical models using expectation maximization. However, this approach has many shortcomings, including a long training time, poor generalizability, and an inability to leverage the inherit semantics of words to determine topics. The method used here remedies many of these issues, by using a learned neural model rather than a statistical model.

## Algorithm Outline
The neural network consists of the following steps:

- Each sentence is split into words and each word is vectorized
- Each vectorized sentence goes through an embedding algorithm to obtain a vector representing the semantics of the sentence as a whole
    - This involves taking a weighted average of the vectors of the words in the sentence, for details see below
- The algorithm attempts to reconstruct a vector similar to the embedding vector only using the topic vectors
    - In this neural topic modelling algorithm, the topics are represented as vectors just as words
- The loss is calculated by comparing the sentence embedding, reconstruction, and the pseudo-embeddings of a small sample of other sentences in the corpus
    - The algorithm learns to combine the topics in a way to closely match the actual embedding, as well as being distinct from embeddings of other sentences

## Sentence Embedding

- The algorithm attempts to get the meaning of a sentence by getting a weighted average of the word vectors in the sentence
- First, a simple average of the word vectors is calculated
    - This is used as a rough estimate of the general context of the sentence
- Then, each word vector is multiplied by the average vector and a learned attention matrix M
    - The output is a single number, representing the prominence the word is judged to have within the sentence
- After a prominence value is calculated for every word, they are passed through a softmax function to get the final proportions of each word in the overall sentence vector

### Performance

- Due to the softmax step, this process has a strong tendency to delegate nearly all the attention to a single word in the sentence
    - This is because even small differences (~1) in the prominance values means the higher word has multiple times more prominance in the final combination
    - Often, this is a good thing, since in many sentences only one word is relevant to the topic
    - However, in many cases multiple words are relevant to the topic/combination of topics
        - In most of these cases the algorithm fails to reflect this reality
    - It is possible to replace softmax with sparsemax (details below), but this does not seem to help, and in fact if anything makes the problem worse

## Sentence Reconstruction

- The algorithm attempts to stratify the sentence into topics by approximating its embedding with topic vectors
- This is done through what is equivalent to a neural network with one softmax layer
- The ultimate goal is to approximate the embedding vector with a combination of topic vectors determined by the neural network, called the sentence reconstruction
    - The loss is calculated over all embeddings, as well as over 20 negative samples for each embedding
    - For each embedding and negative sample, the loss is 1-zr+zn, where:
        - z is the embedding vector
        - r is the reconstruction vector
        - n is the negative sample
        - The dot product is taken between vectors
    - Negative samples are chosen randomly from the corpus
        - The vectors are just simple means of the words vectors in the sentence, rather than true embeddings
        - This done to reduce computation time, as calculating embedding is relatively costly and using embeddings for negative sample would increase the number of calculated embeddings by twentyfold
    - If the loss value is negative, the loss is 0 instead

### Topic Seeding

- In many cases it is desirable to restrict the flexibility of the topics, i.e. seeding
- Some topics can be made into 'seed topics', which, instead of being a fully flexible embedding vector, must be a linear combination of a limited number of seed words
    - The weights of the linear combination are learned similarly to other parameters in the model
        - However, sparsemax is used instead of softmax to avoid having drastic changes in the topic vectors
    - The aim of doing this is to have more comprehensible topics
        - Each seeded topic is to be understood as covering the general semantic area of its seed words, which should relate to a single topic
        - These topics are much easier to analyze than raw vectors
    - It may also be desirable to force the model to use a certain topic, for example if a certain aspect of the corpus is small but one wishes to isolate it

### Performance

- The model fairly easily learns to create very close reconstructions of the embedding vectors using the topic vectors
- However, there are some problems that stem from this:
    - With unseeded topics, the topics are made to match the data very well, but are difficult to interpret
        - The model is likely overfitting the data
    - With seeded topics, learning is slightly slower, but the increase in comprehensibility is large
        - But even with less flexible topic vectors the model can begin to overfit the data, though to a lesser degree
        - One issue is that sometimes the learned weights of the seed words zero out one or more of the words for a topic, making the vector a less representative combination of the words intended for its topic
    - These issues can be fixed by using constant weights
        - However, using a simple mean does not adequately reflect the different amounts the seed words contribute to the topic, and would require much more intervention to make sure the topics are proper
        - Other possibilities will be discussed before, in future steps

## Future Steps

- One of the main areas of possible improvement is in the topic seeding procedure, which will allow the model to create more comprehensible outputs, rather than ones that are merely mathematically optimized
    - As mentioned above, using constant weights for seeded topic vectors can decrease overfitting
    - One possible solution is to use tf-idf on the topics
        - A small portion of the corpus would have to be manually labelled for the desired topics
        - Then a tf-idf analysis would be used to determine the importance of each seed word in how likely a sentence is to be a certain topic
        - These will be translated into weights, with words more indicative of a topic being weighted higher in the corresponding topic

## Source

- The bulk of the algorithm above is from:
    - Angelidis, Stefanos, and Mirella Lapata. "Summarizing opinions: Aspect extraction meets sentiment prediction and they are both weakly supervised." *arXiv preprint arXiv:1808.08858* (2018).