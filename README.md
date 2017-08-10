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
we will see (using Theano as backend. Theano 0.90, Keras: 2.0.6. Macbook Pro 2016. CPU 4 cores. %CPU = 400%. There is no Nvidia GPU for MBP2016):
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
Epoch 1/10 145s - loss: 0.6196 - acc: 0.6223 - val_loss: 0.3755 - val_acc: 0.8437
Epoch 2/10 147s - loss: 0.3605 - acc: 0.8418 - val_loss: 0.2942 - val_acc: 0.8841
Epoch 3/10 150s - loss: 0.2974 - acc: 0.8755 - val_loss: 0.2847 - val_acc: 0.8820
Epoch 4/10 149s - loss: 0.2733 - acc: 0.8872 - val_loss: 0.2662 - val_acc: 0.8928
Epoch 5/10 149s - loss: 0.2505 - acc: 0.8983 - val_loss: 0.2625 - val_acc: 0.8921
Epoch 6/10 154s - loss: 0.2402 - acc: 0.9015 - val_loss: 0.2622 - val_acc: 0.8922
Epoch 7/10 153s - loss: 0.2344 - acc: 0.9055 - val_loss: 0.2626 - val_acc: 0.8913
Epoch 8/10 154s - loss: 0.2197 - acc: 0.9131 - val_loss: 0.2626 - val_acc: 0.8906
Epoch 9/10 153s - loss: 0.2146 - acc: 0.9128 - val_loss: 0.2652 - val_acc: 0.8890
Epoch 10/10 150s - loss: 0.2053 - acc: 0.9186 - val_loss: 0.2659 - val_acc: 0.8905
</pre>

On Ubuntu 16.04, with Python 2.7, Theano 0.90, Keras 2.0.5 (I tried Keras 2.0.2, and turns out it has a bug when loading imdb data)
<pre>
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

Something different at the first few epochs, when running with **python 3.6**, theano 0.90, keras 2.0.5
```bash
CUDA_VISIBLE_DEVICES="1" python sentiment_cnn.py
```
<pre>
Using Theano backend.
Using cuDNN version 6021 on context None
Mapped name None to device cuda0: GeForce GTX 1080 Ti (0000:03:00.0)
Load data...
x_train shape: (25000, 400)
x_test shape: (25000, 400)
Vocabulary Size: 88585
Model type is CNN-non-static
Load existing Word2Vec model '50features_1minwords_10context'
Initializing embedding layer with word2vec weights, shape (88585, 50)
Train on 25000 samples, validate on 25000 samples
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


On Ubuntu 16.04, with Python 3.6, Tensorflow 1.21, Keras 2.0.5
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

Now, on the same Ubuntu 16.04, python 2.7, theano 0.9, Keras 2.0.5, use CPU (only one core, %CPU  = 100%)
```bash
CUDA_VISIBLE_DEVICES="" python sentiment_cnn.py
```
<pre>
Epoch 1/10 220s - loss: 0.6418 - acc: 0.5917 - val_loss: 0.4179 - val_acc: 0.8231
Epoch 2/10 220s - loss: 0.3845 - acc: 0.8299 - val_loss: 0.3086 - val_acc: 0.8724
Epoch 3/10 219s - loss: 0.3085 - acc: 0.8697 - val_loss: 0.3131 - val_acc: 0.8679
Epoch 4/10 219s - loss: 0.2767 - acc: 0.8852 - val_loss: 0.2822 - val_acc: 0.8839
</pre>

Now, on the same Ubuntu 16.04, python 2.7, tensorflow 1.2.1, Keras 2.0.5, use CPU (6 cores, %CPU = 1015% when running `top`),
```bash
CUDA_VISIBLE_DEVICES="" python sentiment_cnn.py
```
<pre>
Epoch 1/10 30s - loss: 0.6273 - acc: 0.6072 - val_loss: 0.3935 - val_acc: 0.8368
Epoch 2/10 30s - loss: 0.3689 - acc: 0.8400 - val_loss: 0.3056 - val_acc: 0.8756
Epoch 3/10 30s - loss: 0.3032 - acc: 0.8741 - val_loss: 0.2908 - val_acc: 0.8822
Epoch 4/10 30s - loss: 0.2734 - acc: 0.8883 - val_loss: 0.2747 - val_acc: 0.8889
Epoch 5/10 30s - loss: 0.2555 - acc: 0.8966 - val_loss: 0.3084 - val_acc: 0.8652
Epoch 6/10 30s - loss: 0.2423 - acc: 0.9041 - val_loss: 0.2698 - val_acc: 0.8867
Epoch 7/10 30s - loss: 0.2330 - acc: 0.9060 - val_loss: 0.2717 - val_acc: 0.8874
Epoch 8/10 30s - loss: 0.2235 - acc: 0.9112 - val_loss: 0.2846 - val_acc: 0.8808
Epoch 9/10 30s - loss: 0.2154 - acc: 0.9138 - val_loss: 0.2758 - val_acc: 0.8858
Epoch 10/10 30s - loss: 0.2093 - acc: 0.9148 - val_loss: 0.2743 - val_acc: 0.8862
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
