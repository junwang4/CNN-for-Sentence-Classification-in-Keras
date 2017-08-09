# Convolutional Neural Networks for Sentence Classification

Train convolutional network for sentiment analysis. Based on "Convolutional Neural Networks for Sentence Classification" by Yoon Kim, [link](http://arxiv.org/pdf/1408.5882v2.pdf). Inspired by Denny Britz article "Implementing a CNN for Text Classification in TensorFlow", [link](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/).
For "CNN-rand" and "CNN-non-static" gets to 88-90%, and "CNN-static" - 85%

## Some difference from original article:
* larger IMDB corpus, longer sentences; sentence length is very important, just like data size
* smaller embedding dimension, 20 instead of 300
* 2 filter sizes instead of original 3
* much fewer filters; experiments show that 3-10 is enough; original work uses 100
* random initialization is no worse than word2vec init on IMDB corpus
* sliding Max Pooling instead of original Global Pooling

## Dependencies

* The [Keras](http://keras.io/) Deep Learning library and most recent [Theano](http://deeplearning.net/software/theano/install.html#install) backend should be installed. You can use pip for that. 
Not tested with TensorFlow, but should work.

## NOTE
In the original paper by Kim, the word embeddings are from [GoogleNews-vectors-negative300.bin](https://code.google.com/archive/p/word2vec/).
But in [Alexander Rakhli's Keras-based implementation](https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras),
the word embeddings are generated on the fly using the Gensim.
To replicate Kim's result, this fork will use the GoogleNews-vectors-negative300.bin
