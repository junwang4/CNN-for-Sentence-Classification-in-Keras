# Convolutional Neural Networks for Sentence Classification

## Introduction
- Forked from [Alexander Rakhli's Keras-based implementation](https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras), which is
- Based on Yoon Kim's EMNLP 2014's paper and code: [Convolutional Neural Networks for Sentence Classification](https://github.com/yoonkim/CNN_sentence), 
- Inspired by Denny Britz's blog article [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/).

## Motivation
- Kim's version is based on Theano version 0.7, which is now outdated (for example, backend changed from `device=gpu` to `device=cuda`). I tried to create a virtual env for python 2.7 and theano 0.7, but it just doesn't work for GPU, and I think that may be related to the gpuarray as the new gpu backend. 
- In the original work by Kim, the pretrained word vectors are [GoogleNews-vectors-negative300.bin](https://code.google.com/archive/p/word2vec/).
But in [Alexander Rakhli's Keras-based implementation](https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras),
the word vectors are generated on the fly using the [Gensim](https://radimrehurek.com/gensim/).
To replicate Kim's result, this fork will use the GoogleNews word vectors.

## Usage

### Running the Stanford IMDB 50,000 movie reviews dataset (25000 for train, and 25000 for test)
Default options:
```python
model_type = "CNN-non-static"  # CNN-rand|CNN-non-static|CNN-static

data_source = "keras_data_set"  # keras_data_set|local_dir

embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

# Training parameters
batch_size = 64
num_epochs = 10

# Prepossessing parameters
sequence_length = 400
max_words = 5000

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10
```

Then run
```bash
$ python sentiment_cnn.py
```
we will see (using Theano as backend. My version Theano 0.90. Macbook Pro 2016. CPU.):
Using Theano backend.
<pre>
Load data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb_word_index.json
1343488/1641221 [=======================>......] - ETA: 0s('x_train shape:', (25000, 400))
('x_test shape:', (25000, 400))
Vocabulary Size: 88585
('Model type is', 'CNN-non-static')
Training Word2Vec model...
Saving Word2Vec model '50features_1minwords_10context'
('Initializing embedding layer with word2vec weights, shape', (88585, 50))
Train on 25000 samples, validate on 25000 samples
Epoch 1/10
145s - loss: 0.6196 - acc: 0.6223 - val_loss: 0.3755 - val_acc: 0.8437
</pre>



### Running the local data (Cornell Movie Review Dataset) with Gensim-generated word vectors
- modify `sentiment_cnn.py`
```python
data_source = "local_dir"
```

```bash
$ python sentiment_cnn.py
```

### Running the above data with pretrained GoogleNews w2v



---
## Below: Copied from 
[Alexander Rakhli's Keras-based implementation](https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras),

For "CNN-rand" and "CNN-non-static" gets to 88-90%, and "CNN-static" - 85%
### Some difference from original article:
* larger IMDB corpus, longer sentences; sentence length is very important, just like data size
* smaller embedding dimension, 20 instead of 300
* 2 filter sizes instead of original 3
* much fewer filters; experiments show that 3-10 is enough; original work uses 100
* random initialization is no worse than word2vec init on IMDB corpus
* sliding Max Pooling instead of original Global Pooling

### Dependencies

* The [Keras](http://keras.io/) Deep Learning library and most recent [Theano](http://deeplearning.net/software/theano/install.html#install) backend should be installed. You can use pip for that. 
Not tested with TensorFlow, but should work.
