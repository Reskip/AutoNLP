import os
import gzip
import re
import jieba
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.optimizers import adam
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.python.keras.preprocessing import text
from tensorflow.python.keras.preprocessing import sequence

from cnn import cnn_model

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
# to log device placement (on which device the operation ran)
config.log_device_placement = True
# (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
# set this TensorFlow session as the default session for Keras
set_session(sess)

IS_TEST = False
MAX_SEQ_LENGTH = 500
MAX_VOCAB_SIZE = 20000  # Limit the number of features. only top 20K features


def clean_en_text(dat):

    REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')

    ret = []
    for line in dat:
        # text = text.lower() # lowercase text
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = BAD_SYMBOLS_RE.sub('', line)
        line = line.strip()
        ret.append(line)
    return ret


def clean_zh_text(dat):
    REPLACE_BY_SPACE_RE = re.compile('[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')

    ret = []
    for line in dat:
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = line.strip()
        ret.append(line)
    return ret


def sequentialize_data(train_contents, val_contents=None):
    """Vectorize data into ngram vectors.

    Args:
        train_contents: training instances
        val_contents: validation instances
        y_train: labels of train data.

    Returns:
        sparse ngram vectors of train, valid text inputs.
    """
    tokenizer = text.Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(train_contents)
    x_train = tokenizer.texts_to_sequences(train_contents)

    if val_contents:
        x_val = tokenizer.texts_to_sequences(val_contents)

    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQ_LENGTH:
        max_length = MAX_SEQ_LENGTH

    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    if val_contents:
        x_val = sequence.pad_sequences(x_val, maxlen=max_length)

    word_index = tokenizer.word_index
    num_features = min(len(word_index) + 1, MAX_VOCAB_SIZE)
    if val_contents:
        return x_train, x_val, word_index, num_features, tokenizer, max_length
    else:
        return x_train, word_index, num_features, tokenizer, max_length


def _tokenize_chinese_words(text):
    return ' '.join(jieba.cut(text, cut_all=False))


def vectorize_data(x_train, x_val=None):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    if x_val:
        full_text = x_train + x_val
    else:
        full_text = x_train
    vectorizer.fit(full_text)
    train_vectorized = vectorizer.transform(x_train)
    if x_val:
        val_vectorized = vectorizer.transform(x_val)
        return train_vectorized, val_vectorized, vectorizer
    return train_vectorized, vectorizer


# onhot encode to category
def ohe2cat(label):
    return np.argmax(label, axis=1)


class Model(object):
    """ 
        model of CNN baseline without pretraining.
        see `https://aclweb.org/anthology/D14-1181` for more information.
    """

    def __init__(self, metadata, train_output_path="./", test_input_path="./"):
        """ Initialization for model
        :param metadata: a dict formed like:
            {"class_num": 10,
             "language": ZH,
             "num_train_instances": 10000,
             "num_test_instances": 1000,
             "time_budget": 300}
        """
        self.done_training = False
        self.metadata = metadata
        self.train_output_path = train_output_path
        self.test_input_path = test_input_path

        # user model variable
        self.model = None
        self.tokenizer = None
        self.max_length = None
        self.x_train = None
        self.y_train = None
        self.word_index = None
        self.num_features = None
        self.num_classes = None
        self.embedding_matrix = None

        self.initialized = False

    def initialize(self, train_dataset):
        # why not init in __init__() ?
        # cause word index process need train_dataset, so init step should run
        # in first training step

        self.initialized = True
        self.x_train, self.y_train = train_dataset

        # tokenize Chinese words
        if self.metadata['language'] == 'ZH':
            self.x_train = clean_zh_text(self.x_train)
            self.x_train = list(map(_tokenize_chinese_words, self.x_train))
        else:
            self.x_train = clean_en_text(self.x_train)

        self.x_train, self.word_index, self.num_features, self.tokenizer,\
            self.max_length = sequentialize_data(self.x_train)
        self.num_classes = self.metadata['class_num']

        # loading pretrained embedding
        FT_DIR = '/app/embedding'
        fasttext_embeddings_index = {}
        if self.metadata['language'] == 'ZH':
            f = gzip.open(os.path.join(FT_DIR, 'cc.zh.300.vec.gz'), 'rb')
        elif self.metadata['language'] == 'EN':
            f = gzip.open(os.path.join(FT_DIR, 'cc.en.300.vec.gz'), 'rb')
        else:
            raise ValueError('Unexpected embedding path:'
                             ' {unexpected_embedding}. '.format(
                                 unexpected_embedding=FT_DIR))

        LOCALE_TEST_WORD_LIMIT = 50000
        for line in f.readlines():
            if IS_TEST and LOCALE_TEST_WORD_LIMIT <= 0:
                break
            LOCALE_TEST_WORD_LIMIT -= 1

            values = line.strip().split()
            if self.metadata['language'] == 'ZH':
                word = values[0].decode('utf8')
            else:
                word = bytes.decode(values[0])
            coefs = np.asarray(values[1:], dtype='float32')
            fasttext_embeddings_index[word] = coefs

        print('Found %s fastText word vectors.' %
              len(fasttext_embeddings_index))
        # embedding lookup
        EMBEDDING_DIM = 300
        self.embedding_matrix = np.zeros((self.num_features, EMBEDDING_DIM))
        cnt = 0
        correct_cnt = 0
        for word, i in self.word_index.items():
            if i >= self.num_features:
                continue
            embedding_vector = fasttext_embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector
                correct_cnt += 1
            else:
                self.embedding_matrix[i] = np.zeros(300)
                cnt += 1

        self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
        print('fastText oov words: %s, correct words: %s' % (cnt, correct_cnt))

    def init_model(self):
        # initialize model
        self.model = cnn_model(
            input_shape=self.x_train.shape[1:][0],
            num_classes=self.num_classes,
            num_features=self.num_features,
            embedding_matrix=self.embedding_matrix,
            filters=128,
            kernel_sizes=[2, 3, 4, 5],
            dropout_rate=0.4,
            embedding_trainable=True,
            l2_lambda=1.0)

        loss = 'sparse_categorical_crossentropy'
        optimizer = adam(lr=1e-3)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    def train(self, train_dataset, remaining_time_budget=None):
        """model training on train_dataset.

        :param train_dataset: tuple, (x_train, y_train)
            x_train: list of str, input training sentences.
            y_train: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        :param remaining_time_budget:
        """
        if not self.initialized:
            self.initialize(train_dataset)

        if self.done_training:
            return

        if self.model is None:
            self.init_model()

        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3)]

        # fit model
        history = self.model.fit(
            self.x_train,
            ohe2cat(self.y_train),
            # y_train,
            epochs=5,
            callbacks=callbacks,
            validation_split=0.2,
            # validation_data=(x_dev,y_dev),
            verbose=2,  # Logs once per epoch.
            batch_size=32,
            shuffle=False)
        print(str(type(self.x_train)) + " " + str(self.y_train.shape))
        print(history.history)
        if len(history.history['loss']) < 5:
            self.done_training = True

    def test(self, x_test, remaining_time_budget=None):
        """
        :param x_test: list of str, input test sentences.
        :param remaining_time_budget:
        :return: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                 here `sample_count` is the number of examples in this dataset as test
                 set and `class_num` is the same as the class_num in metadata. The
                 values should be binary or in the interval [0,1].
        """
        # tokenizing Chinese words
        if self.metadata['language'] == 'ZH':
            x_test = clean_zh_text(x_test)
            x_test = list(map(_tokenize_chinese_words, x_test))
        else:
            x_test = clean_en_text(x_test)

        x_test = self.tokenizer.texts_to_sequences(x_test)
        x_test = sequence.pad_sequences(x_test, maxlen=self.max_length)
        result = self.model.predict(x_test)
        print(result[0])

        return result
