import os
import gzip
import re
import jieba_fast as jieba
import tensorflow as tf
import numpy as np
import time
import math
from tensorflow.python.keras.optimizers import adam
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing import text
from tensorflow.python.keras.preprocessing import sequence

from cnn import cnn_model
from timer import Timer, _Timer

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

MAX_SEQ_LENGTH = 500
MAX_VOCAB_SIZE = 20000  # Limit the number of features. only top 20K features
MAX_VOCAB_READ_FROM_EMBEDDING = 500000  # Limit number of wordvec load from dict
MAX_VOCAB_STAT = 50000
BATCH_SIZE = 1000
MAX_VALID_SIZE = 2000
MAX_TRAIN_TIME = 240
MIN_FIRST_TRAIN = 15


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


def user_log(text):
    sp = time.time()
    localtime = time.localtime(sp)
    print("%s [USER LOG] %s" % (time.strftime("%Y-%m-%d %H:%M:%S", localtime), text))


class Model(object):
    """
        model of CNN baseline without pretraining.
        see `https://aclweb.org/anthology/D14-1181` for more information.
    """

    def load_word_vec(self):
        self._timer.start("load_word_vec")
        # loading pretrained embedding
        FT_DIR = '/app/embedding'

        if self.metadata['language'] == 'ZH':
            f = gzip.open(os.path.join(FT_DIR, 'cc.zh.300.vec.gz'), 'rb')
        elif self.metadata['language'] == 'EN':
            f = gzip.open(os.path.join(FT_DIR, 'cc.en.300.vec.gz'), 'rb')
        else:
            raise ValueError('Unexpected embedding path:'
                             ' {unexpected_embedding}. '.format(
                                 unexpected_embedding=FT_DIR))

        vocab_count = 0
        for line in f:
            if vocab_count > MAX_VOCAB_READ_FROM_EMBEDDING:
                break
            vocab_count += 1

            values = line.strip().split()
            if self.metadata['language'] == 'ZH':
                word = values[0].decode('utf8')
            else:
                word = bytes.decode(values[0])
            coefs = np.asarray(values[1:], dtype='float32')
            self.fasttext_embeddings_index[word] = coefs

        self._timer.end("load_word_vec")
        user_log('Found %s fastText word vectors.' %
                len(self.fasttext_embeddings_index))

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
        self.x_valid = None
        self.y_valid = None
        self.x_test = None
        self.word_index = None
        self.num_features = None
        self.num_classes = None
        self.embedding_matrix = None
        self.fasttext_embeddings_index = dict()

        self.initialized = False
        self.round = 1
        self.trans_x = None
        self.trans_valid = False

        self._now_batch_id = 0
        self._batch_num = 0
        self._loss_history = list()
        self._timer = Timer()
        self._min_loss = 999999999.0
        self._not_imporove_round = 0
        self.load_word_vec()
        user_log("__init__ done.")

    def preprocess(self, i, overwritten=None):
        work_ptr = self.x_train[i]
        if overwritten is not None:
            work_ptr = overwritten

        if self.trans_x[i] is False or overwritten is not None:
            if overwritten is None:
                self.trans_x[i] = True

            self._timer.start("train_transform")
            if self.metadata['language'] == 'ZH':
                work_ptr = clean_zh_text(work_ptr)
                work_ptr = list(map(_tokenize_chinese_words, work_ptr))
            else:
                work_ptr = clean_en_text(work_ptr)

            work_ptr = self.tokenizer.texts_to_sequences(work_ptr)
            work_ptr = sequence.pad_sequences(work_ptr, maxlen=self.max_length)

            if overwritten is None:
                self.x_train[i] = work_ptr
                user_log("Update trans of batch %d." % (i))
            self._timer.end("train_transform")
        return work_ptr

    def initialize(self, train_dataset):
        # why not init in __init__() ?
        # cause word index process need train_dataset, so init step should run
        # in first training step
        user_log("initialize start.")
        self._timer.start("sequentialize")
        self.initialized = True
        self.x_train, self.y_train = train_dataset

        self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
            self.x_train, self.y_train, test_size=0.2)
        self.x_valid = self.x_valid[:MAX_VALID_SIZE]
        self.y_valid = self.y_valid[:MAX_VALID_SIZE]

        self._batch_num = math.ceil(len(self.x_train) / (BATCH_SIZE * 1.0))
        self.trans_x = [False for i in range(self._batch_num)]
        self.x_train = [self.x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE] for i in range(self._batch_num)]
        self.y_train = [self.y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE] for i in range(self._batch_num)]

        x_stat = list()
        index = 0
        while index < self._batch_num and len(x_stat) < MAX_VOCAB_STAT:
            x_stat.extend(self.x_train[index])
            index += 1

        # tokenize Chinese words
        if self.metadata['language'] == 'ZH':
            x_stat = clean_zh_text(x_stat)
            x_stat = list(map(_tokenize_chinese_words, x_stat))
        else:
            x_stat = clean_en_text(x_stat)

        _, self.word_index, self.num_features, self.tokenizer,\
            self.max_length = sequentialize_data(x_stat)
        self.num_classes = self.metadata['class_num']
        self._timer.end("sequentialize")

        self._timer.start("embedding_mapping")
        # embedding lookup
        EMBEDDING_DIM = 300
        self.embedding_matrix = np.zeros((self.num_features, EMBEDDING_DIM))
        cnt = 0
        correct_cnt = 0
        for word, i in self.word_index.items():
            if i >= self.num_features:
                continue
            embedding_vector = self.fasttext_embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector
                correct_cnt += 1
            else:
                self.embedding_matrix[i] = np.zeros(300)
                cnt += 1
        self._timer.end("embedding_mapping")
        user_log('fastText oov words: %s, correct words: %s, split to %d batch.'
            % (cnt, correct_cnt, self._batch_num))

    def init_model(self):
        # initialize model

        self.model = cnn_model(
            input_shape=self.x_train[0].shape[1:][0],
            num_classes=self.num_classes,
            num_features=self.num_features,
            embedding_matrix=self.embedding_matrix,
            filters=64,
            kernel_sizes=[3, 4, 5],
            dropout_rate=0.4,
            embedding_trainable=True,
            l2_lambda=1.0)

        loss = 'sparse_categorical_crossentropy'
        optimizer = adam(lr=1e-3)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    def check_if_end(self):
        if self._not_imporove_round >= 3:
            self.done_training = True

    def validate(self):
        if self.trans_valid is False:
            self.x_valid = self.preprocess(0, self.x_valid)
            self.trans_valid = True

        self._timer.start("validate")
        result = self.model.predict(self.x_valid)
        loss = log_loss(self.y_valid, result)
        self._loss_history.append(loss)
        if loss < self._min_loss:
            self._min_loss = loss
            self._not_imporove_round = 0
        else:
            self._not_imporove_round += 1
        self._timer.end("validate")

    def send_log(self):
        user_log("============================================")
        user_log("Load word vec total time %f." % (self._timer.get("load_word_vec").avg()))
        user_log("Sequentialize total time %f." % (self._timer.get("sequentialize").avg()))
        user_log("Train avg time %f." % (self._timer.get("train_10k").avg()))
        user_log("Test pred avg time %f." % (self._timer.get("test_pred").avg()))
        user_log("Test trans total time %f." % (self._timer.get("test_transform").avg()))
        user_log("Validate avg time %f." % (self._timer.get("validate").avg()))
        user_log("Train trans total time %f." % (self._timer.get("train_transform")._total_time))
        user_log("Mapping embedding total time %f." % (self._timer.get("embedding_mapping")._total_time))
        user_log("Min loss %f." % (self._min_loss))
        user_log("Loss history %s." % (self._loss_history))
        user_log("============================================")

    def run_train(self):
        self.preprocess(self._now_batch_id)
        self._timer.start("train_10k")
        # fit model
        history = self.model.fit(
            self.x_train[self._now_batch_id],
            ohe2cat(self.y_train[self._now_batch_id]),
            # y_train,
            epochs=1,
            verbose=2,  # Logs once per epoch.
            batch_size=32,
            shuffle=False)
        user_log("train batch %d. size %s" % (self._now_batch_id,
            str(self.y_train[self._now_batch_id].shape)))

        self._now_batch_id += 1
        self._now_batch_id %= self._batch_num
        user_log("history %s." % (history.history))

        self._timer.end("train_10k")

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
        user_log("train_start. remain %f sec" % (remaining_time_budget))
        max_train_time = min(
            MIN_FIRST_TRAIN * self.round,
            5*self._timer.get("test_pred").avg() if self._timer.get("test_pred").avg() > 0 else 99999999,
            MAX_TRAIN_TIME)
        self.round += 1

        if not self.initialized:
            self.initialize(train_dataset)

        if self.done_training:
            return

        if self.model is None:
            self.preprocess(0)
            self.init_model()

        train_runtime_timer = _Timer()
        while train_runtime_timer._total_time < max_train_time:
            train_runtime_timer.start()
            self.run_train()
            train_runtime_timer.end()

        user_log("Train finished. train_time_budget: %f, actually train: %f." % (
            max_train_time,
            train_runtime_timer._total_time
        ))

        if self.round > 2:
            self.validate()
            self.check_if_end()

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
        if self.x_test is None:
            self._timer.start("test_transform")
            if self.metadata['language'] == 'ZH':
                self.x_test = clean_zh_text(x_test)
                self.x_test = list(map(_tokenize_chinese_words, self.x_test))
            else:
                self.x_test = clean_en_text(x_test)

            self.x_test = self.tokenizer.texts_to_sequences(self.x_test)
            self.x_test = sequence.pad_sequences(self.x_test, maxlen=self.max_length)
            self._timer.end("test_transform")

        self._timer.start("test_pred")
        result = self.model.predict(self.x_test)
        user_log(result[0])
        self._timer.end("test_pred")

        self.send_log()
        return result
