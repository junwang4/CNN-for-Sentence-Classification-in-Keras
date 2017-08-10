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
### Results

Settings:
- GPU: GTX 1080 Ti
- CPU: i7-6850K
- Ubuntu 16.04
- Nvidia Driver 381.22, CUDA 8.0, cuDNN 6.0
- Keras 2.0.5 (I tried Keras 2.0.2, and turns out it has a bug when loading IMDB data)
- Tensorflow 1.2
- Theano 0.9  (which is about 3x faster than tensorflow !!!)
- Python 2.7 or 3.6

#### Python 2.7 / Theano
<pre>
Load data...
x_train shape: (25000, 400)
x_test shape: (25000, 400)
Vocabulary Size: 88585
Model type is CNN-non-static
Load existing Word2Vec model '50features_1minwords_10context'
Train on 25000 samples, validate on 25000 samples
Epoch 1/10 2s - loss: 0.6406 - acc: 0.5952 - val_loss: 0.4204 - val_acc: 0.8214
Epoch 2/10 1s - loss: 0.3822 - acc: 0.8342 - val_loss: 0.3098 - val_acc: 0.8704
Epoch 3/10 1s - loss: 0.3071 - acc: 0.8703 - val_loss: 0.3107 - val_acc: 0.8696
Epoch 4/10 1s - loss: 0.2758 - acc: 0.8853 - val_loss: 0.2817 - val_acc: 0.8846
Epoch 5/10 1s - loss: 0.2556 - acc: 0.8941 - val_loss: 0.2910 - val_acc: 0.8750
Epoch 6/10 1s - loss: 0.2437 - acc: 0.9003 - val_loss: 0.2750 - val_acc: 0.8856
Epoch 7/10 1s - loss: 0.2313 - acc: 0.9087 - val_loss: 0.2749 - val_acc: 0.8848
Epoch 8/10 1s - loss: 0.2254 - acc: 0.9082 - val_loss: 0.2800 - val_acc: 0.8821
Epoch 9/10 1s - loss: 0.2182 - acc: 0.9135 - val_loss: 0.2774 - val_acc: 0.8840
Epoch 10/10 1s - loss: 0.2087 - acc: 0.9170 - val_loss: 0.2784 - val_acc: 0.8847
</pre>

#### Python 3.6 / Theano
- Note that something different at the first few epochs, when running with **python 3.6**
- This is also true for Tensorflow when using python 3.6
<pre>
Epoch 1/10 1s - loss: 0.6935 - acc: 0.4980 - val_loss: 0.6932 - val_acc: 0.5000
Epoch 2/10 1s - loss: 0.6932 - acc: 0.5028 - val_loss: 0.6929 - val_acc: 0.5159
Epoch 3/10 1s - loss: 0.6891 - acc: 0.5331 - val_loss: 0.6791 - val_acc: 0.5936
Epoch 4/10 1s - loss: 0.5206 - acc: 0.7351 - val_loss: 0.3486 - val_acc: 0.8596
Epoch 5/10 1s - loss: 0.3530 - acc: 0.8493 - val_loss: 0.3337 - val_acc: 0.8500
Epoch 6/10 1s - loss: 0.3140 - acc: 0.8687 - val_loss: 0.2831 - val_acc: 0.8863
Epoch 7/10 1s - loss: 0.2954 - acc: 0.8775 - val_loss: 0.2835 - val_acc: 0.8842
Epoch 8/10 1s - loss: 0.2761 - acc: 0.8870 - val_loss: 0.2747 - val_acc: 0.8858
Epoch 9/10 1s - loss: 0.2615 - acc: 0.8928 - val_loss: 0.2787 - val_acc: 0.8822
Epoch 10/10 1s - loss: 0.2527 - acc: 0.8962 - val_loss: 0.2711 - val_acc: 0.8888
</pre>


#### Python 2.7 / Tensorflow
<pre>
Epoch 1/10 5s - loss: 0.6309 - acc: 0.6059 - val_loss: 0.3901 - val_acc: 0.8381
Epoch 2/10 4s - loss: 0.3761 - acc: 0.8351 - val_loss: 0.3074 - val_acc: 0.8743
Epoch 3/10 4s - loss: 0.3038 - acc: 0.8727 - val_loss: 0.2855 - val_acc: 0.8846
Epoch 4/10 4s - loss: 0.2765 - acc: 0.8865 - val_loss: 0.2761 - val_acc: 0.8871
Epoch 5/10 4s - loss: 0.2569 - acc: 0.8968 - val_loss: 0.3076 - val_acc: 0.8638
Epoch 6/10 4s - loss: 0.2429 - acc: 0.9014 - val_loss: 0.2686 - val_acc: 0.8878
Epoch 7/10 4s - loss: 0.2294 - acc: 0.9067 - val_loss: 0.2683 - val_acc: 0.8888
Epoch 8/10 4s - loss: 0.2230 - acc: 0.9091 - val_loss: 0.2751 - val_acc: 0.8844
Epoch 9/10 4s - loss: 0.2114 - acc: 0.9138 - val_loss: 0.2740 - val_acc: 0.8859
Epoch 10/10 4s - loss: 0.2069 - acc: 0.9172 - val_loss: 0.2726 - val_acc: 0.8874
</pre>

#### Python 3.6 / Tensorflow
<pre>
Epoch 1/10 5s - loss: 0.6936 - acc: 0.5044 - val_loss: 0.6929 - val_acc: 0.5106
Epoch 2/10 4s - loss: 0.6925 - acc: 0.5130 - val_loss: 0.6888 - val_acc: 0.5500
Epoch 3/10 4s - loss: 0.5746 - acc: 0.6787 - val_loss: 0.3575 - val_acc: 0.8535
Epoch 4/10 4s - loss: 0.3743 - acc: 0.8380 - val_loss: 0.3058 - val_acc: 0.8775
Epoch 5/10 4s - loss: 0.3157 - acc: 0.8678 - val_loss: 0.3039 - val_acc: 0.8711
Epoch 6/10 4s - loss: 0.3021 - acc: 0.8722 - val_loss: 0.2795 - val_acc: 0.8863
Epoch 7/10 4s - loss: 0.2783 - acc: 0.8856 - val_loss: 0.2779 - val_acc: 0.8873
Epoch 8/10 4s - loss: 0.2648 - acc: 0.8921 - val_loss: 0.2720 - val_acc: 0.8880
Epoch 9/10 4s - loss: 0.2571 - acc: 0.8914 - val_loss: 0.2744 - val_acc: 0.8885
Epoch 10/10 4s - loss: 0.2434 - acc: 0.9008 - val_loss: 0.2744 - val_acc: 0.8862
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
