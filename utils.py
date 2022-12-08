import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import re
import string
import spacy
import gensim.downloader as gen

from tqdm import tqdm
import nltk
# nltk.download('all')
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from tensorflow.keras.layers import TextVectorization
from SkipGram import skip_gram
from transformers import AutoTokenizer, TFAutoModel
from unicodedata import normalize



def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class Helper:

    def __init__(self, df_titles, train_df, seq_len, embedding_type, embeded_vector_size):
        self.df_titles = df_titles
        self.train_df = train_df
        self.seq_len = seq_len
        self.embedding_type = embedding_type
        self.embeded_vector_size = embeded_vector_size
        self.embedding_matrix = 'uniform'
        self.vocab_to_index = None
        self.index_to_vocab = None
        self.str_dict_y = None
        self.vectorizer = None

    def tag_to_index(self, y):
        str_list = list(np.unique(y))
        str_dict = {}
        for i, item in enumerate(str_list):
            str_dict[i] = str(item)
            y = np.where(y == item, i, y)
        return y, str_dict

    def vocab_index(self, vocabs):
        idx_to_vocab = {}
        vocab_to_idx = {}

        number_exp = '[0-9]+'
        char_number_exp = '([a-z]+[0-9]+)+([0-9|a-z])*'
        number_char_exp = '([0-9]+[a-z]+)+([0-9|a-z])*'
        for i, vocab in enumerate(vocabs):
            vocab = self.word_cleaning(vocab)
            if vocab != '<pad>':
                vocab = vocab.encode("ascii", "ignore")
                vocab = vocab.decode()
                vocab = "".join([char for char in vocab if char not in string.punctuation])
                if re.fullmatch(number_exp, vocab):
                    vocabs[i] = 'numberexp'
                elif re.fullmatch(char_number_exp, vocab):
                    vocabs[i] = 'charnumberexp'
                elif re.fullmatch(number_char_exp, vocab):
                    vocabs[i] = 'numbercharexp'
                else:
                    vocabs[i] = vocab
            else:
                vocabs[i] = vocab

        vocabs = list(np.unique(vocabs))
        for i, vocab in enumerate(vocabs):
            idx_to_vocab[i] = vocab
            vocab_to_idx[vocab] = i

        self.vocab_to_index = vocab_to_idx
        self.index_to_vocab = idx_to_vocab

    def vocab_builder(self):
        titles = self.df_titles['Title'][30000:2000000]
        words = set()
        for title in tqdm(titles):
            words = words.union(set(title.split(" ")))

        words = list(words)
        words.append('<pad>')
        words.append('unknown')
        words.append('none')
        with open('Data/words_list', 'wb') as fp:
            pickle.dump(words, fp)
            print('Done writing words into a binary file')

    lp = spacy.load("en_core_web_sm")

    def word_cleaning(self, d):
        wordnet_lemmatizer = WordNetLemmatizer()
        stemmer = SnowballStemmer('english', ignore_stopwords=True)
        d = wordnet_lemmatizer.lemmatize(d)
        d = stemmer.stem(d)
        return d

    def clean_data(self, df, column):
        clean_df = df.copy()

        def clean(text):

            text = normalize("NFKD", text)

            return text

        clean_df[column] = clean_df[column].apply(clean)
        clean_df[column] = clean_df[column].apply(self.word_cleaning)

        return clean_df

    def load_data(self):
        x = list(self.train_df['Token'])
        y = self.train_df['Tag']

        y, self.str_dict_y = self.tag_to_index(y)

        x_train = []
        y_train = []

        if self.embedding_type == 'scratch':
            with open('Data/words_list', 'rb') as fp:
                words = pickle.load(fp)
            self.vocab_index(words)
            number_exp = '[0-9]+'
            char_number_exp = '([a-z]+[0-9]+)+([0-9|a-z])*'
            number_char_exp = '([0-9]+[a-z]+)+([0-9|a-z])*'
            for i in range(int(len(x) / 59)):
                x_temp = np.zeros(shape=(59,), dtype=np.int32)
                for j, item in enumerate(x[59 * i:59 * (i + 1)]):
                    item = self.word_cleaning(item)
                    if item != '<pad>':
                        item = "".join(char for char in item if char not in string.punctuation)
                        if re.fullmatch(number_exp, item):
                            x_temp[j] = self.vocab_to_index['numberexp']
                        elif re.fullmatch(char_number_exp, item):
                            x_temp[j] = self.vocab_to_index['charnumberexp']
                        elif re.fullmatch(number_char_exp, item):
                            x_temp[j] = self.vocab_to_index['numbercharexp']
                        else:
                            try:
                                x_temp[j] = self.vocab_to_index[item]
                            except:
                                x_temp[j] = self.vocab_to_index['unknown']
                    else:
                        x_temp[j] = self.vocab_to_index[item]

                x_train.append(x_temp)
                y_train.append(y[59 * i:59 * (i + 1)])

            path_to_glove_file = f'Data/glove.6B/glove.6B.{self.embeded_vector_size}d.txt'
            embeddings_index = {}
            with open(path_to_glove_file) as f:
                for line in f:
                    word, coefs = line.split(maxsplit=1)
                    coefs = np.fromstring(coefs, "f", sep=" ")
                    embeddings_index[word] = coefs

            print("Found %s word vectors." % len(embeddings_index))

            num_vocab = len(self.vocab_to_index)
            embedding_dim = self.embeded_vector_size
            hits = 0
            misses = 0
            # Prepare embedding matrix
            embedding_matrix = np.zeros((num_vocab, embedding_dim))
            for word, i in self.vocab_to_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # Words not found in embedding index will be all-zeros.
                    # This includes the representation for "padding" and "OOV"
                    embedding_matrix[i] = embedding_vector
                    hits += 1
                else:
                    misses += 1
            print("Converted %d words (%d misses)" % (hits, misses))
            self.embedding_matrix = embedding_matrix

            x_train = np.array(x_train)
            y_train = np.array(y_train)
            y_train = tf.keras.utils.to_categorical(y_train, len(self.str_dict_y))

        def custom_standardization(input_data):
            lowercase = tf.strings.lower(input_data)
            return tf.strings.regex_replace(lowercase,
                                            '[%s]' % re.escape(string.punctuation), '')

        if self.embedding_type == 'glove':
            self.vectorizer = TextVectorization(standardize=custom_standardization,
                                                output_sequence_length=59)
            df_titles = self.clean_data(self.df_titles, 'Title')
            text_ds = tf.data.Dataset.from_tensor_slices(df_titles['Title']).batch(1024)
            self.vectorizer.adapt(text_ds)
            voc = self.vectorizer.get_vocabulary()
            print(voc[:5])
            word_index = dict(zip(voc, range(len(voc))))
            self.vocab_to_index = word_index

            path_to_glove_file = f'Data/glove.6B/glove.6B.{self.embeded_vector_size}d.txt'

            embeddings_index = {}
            with open(path_to_glove_file) as f:
                for line in f:
                    word, coefs = line.split(maxsplit=1)
                    coefs = np.fromstring(coefs, "f", sep=" ")
                    embeddings_index[word] = coefs

            print("Found %s word vectors." % len(embeddings_index))

            num_vocab = len(voc)
            embedding_dim = self.embeded_vector_size
            hits = 0
            misses = 0
            # Prepare embedding matrix
            embedding_matrix = np.zeros((num_vocab, embedding_dim))
            for word, i in word_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # Words not found in embedding index will be all-zeros.
                    # This includes the representation for "padding" and "OOV"
                    embedding_matrix[i] = embedding_vector
                    hits += 1
                else:
                    misses += 1
            print("Converted %d words (%d misses)" % (hits, misses))
            self.embedding_matrix = embedding_matrix
            x_train = self.vectorizer(np.array([[s] for s in df_titles['Title'][:5000]])).numpy()

            for i in range(int(len(y) / 59)):
                y_train.append(y[59 * i:59 * (i + 1)])

            x_train = np.array(x_train)
            y_train = np.array(y_train)
            y_train = tf.keras.utils.to_categorical(y_train, len(self.str_dict_y))

        if self.embedding_type == 'skip_gram':
            self.vectorizer = TextVectorization(standardize=custom_standardization,
                                                output_sequence_length=59)
            df_titles = self.clean_data(self.df_titles, 'Title')
            text_ds = tf.data.Dataset.from_tensor_slices(df_titles['Title']).batch(1024)
            self.vectorizer.adapt(text_ds)

            voc = self.vectorizer.get_vocabulary()
            print(voc[:5])
            word_index = dict(zip(voc, range(len(voc))))
            self.vocab_to_index = word_index
            self.index_to_vocab = {index: token for token, index in self.vocab_to_index.items()}
            # Vectorize the data in text_ds.
            text_vector_ds = text_ds.prefetch(tf.data.AUTOTUNE).map(self.vectorizer).unbatch()
            sequences = list(text_vector_ds.as_numpy_iterator())

            skip = skip_gram(sequences,
                                 len(self.vocab_to_index),
                                 self.vocab_to_index,
                                 embedding_dim=self.embeded_vector_size,
                                 window_size=2,
                                 num_ns=4)
            word2vec_vectors = gen.load('word2vec-google-news-300')
            word2vec_vectors = word2vec_vectors.wv
            num_vocab = len(voc)
            embedding_dim = self.embeded_vector_size
            hits = 0
            misses = 0
            # Prepare embedding matrix
            embedding_matrix = np.zeros((num_vocab, embedding_dim))
            for word, i in word_index.items():
                embedding_vector = word2vec_vectors[word]
                if embedding_vector is not None:
                    # Words not found in embedding index will be all-zeros.
                    # This includes the representation for "padding" and "OOV"
                    embedding_matrix[i] = embedding_vector
                    hits += 1
                else:
                    misses += 1
            print("Converted %d words (%d misses)" % (hits, misses))
            self.embedding_matrix = embedding_matrix
            skip.word2vec.get_layer('w2v_embedding').set_weights(self.embedding_matrix)
            word2vec_scratch = skip.word2vec_training()
            self.embedding_matrix = word2vec_scratch.get_layer('w2v_embedding').get_weights()[0]
            x_train = np.array(sequences[:5000])

            for i in range(int(len(y) / 59)):
                y_train.append(y[59 * i:59 * (i + 1)])
            y_train = np.array(y_train)
            y_train = tf.keras.utils.to_categorical(y_train, len(self.str_dict_y))

        if self.embedding_type == 'bert':
            self.vectorizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            df_titles = self.clean_data(self.df_titles, 'Title')
            tokenized_data = self.vectorizer(df_titles['Title'][:5000].to_list(),
                                      padding='max_length',
                                      max_length=90,
                                      return_tensors="tf")

            x_train = tokenized_data['input_ids']
            attention_mask = tokenized_data['attention_mask']
            self.vocab_to_index = self.vectorizer.vocab
            self.embedding_matrix = TFAutoModel.from_pretrained("bert-base-uncased", trainable=False).weights[0]
            self.embeded_vector_size = 768
            self.seq_len = 90

            for i in range(int(len(y) / 59)):
                y_train.append(list(y[59 * i:59 * (i + 1)]) + (self.seq_len - 59)*[33])
            y_train = np.array(y_train)
            y_train = tf.keras.utils.to_categorical(y_train, len(self.str_dict_y))
            x_train = [x_train, attention_mask]

        return x_train, y_train

    def build_dataset(self):
        max_spaces = max(title.count(" ") for title in self.df_titles['Title'])
        print(max_spaces)

        for i in range(5000):
            space_number = self.df_titles['Title'][i].count(" ")
            for j in range(max_spaces - space_number):
                row = pd.DataFrame(
                    {"Record Number": self.train_df["Record Number"][(max_spaces + 1) * i + space_number + j],
                     "Title": self.train_df['Title'][(max_spaces + 1) * i + space_number + j],
                     "Token": "none", "Tag": "none"}, index=[(max_spaces + 1) * i + space_number + j + 1])
                self.train_df = pd.concat([self.train_df.iloc[:(max_spaces + 1) * i + space_number + j + 1], row,
                                             self.train_df.iloc[
                                             (max_spaces + 1) * i + space_number + j + 1:]]).reset_index(drop=True)

        self.train_df.to_pickle('Data/dataset.pkl')

    def test_model(self, model, data):
        if self.embedding_type == 'scratch':
            with open('output.tsv', 'w', encoding='utf-8') as fp:
                for j, row in data.iterrows():
                    title = row['Title']
                    space_number = title.count(" ")
                    for k in range((self.seq_len) - space_number - 1):
                        title += ' none'

                    idxs_title = []
                    number_exp = '[0-9]+'
                    char_number_exp = '([a-z]+[0-9]+)+([0-9|a-z])*'
                    number_char_exp = '([0-9]+[a-z]+)+([0-9|a-z])*'
                    for st in title.split(" "):
                        st = self.word_cleaning(st)
                        if st != '<pad>':
                            st = "".join([char for char in st if char not in string.punctuation])
                            if re.fullmatch(number_exp, st):
                                idxs_title.append(self.vocab_to_index['numberexp'])
                            elif re.fullmatch(char_number_exp, st):
                                idxs_title.append(self.vocab_to_index['charnumberexp'])
                            elif re.fullmatch(number_char_exp, st):
                                idxs_title.append(self.vocab_to_index['numbercharexp'])
                            else:
                                try:
                                    idxs_title.append(self.vocab_to_index[st])
                                except:
                                    idxs_title.append(self.vocab_to_index['unknown'])
                        else:
                            idxs_title.append(self.vocab_to_index[st])

                    idxs_title = np.expand_dims(idxs_title, 0)
                    prediction = model.predict(idxs_title)
                    predicted_idxs = np.argmax(np.squeeze(prediction), axis=1)
                    Aspect_Name = []
                    Aspect_Value = []
                    for st, idx in zip(title.split(" "), predicted_idxs):
                        if st == 'none':
                            break
                        if self.str_dict_y[idx] not in ['none', '']:
                            Aspect_Name.append(self.str_dict_y[idx])
                            Aspect_Value.append(st)
                        elif self.str_dict_y[idx] == '' and len(Aspect_Value) > 0:
                            print(len(Aspect_Value))
                            Aspect_Value[-1] += ' ' + st
                        else:
                            Aspect_Value.append(st)

                    for i in range(len(Aspect_Name)):
                        fp.write(str(row['Record Number']) + "\t" + Aspect_Name[i] + "\t" + Aspect_Value[i] + '\r')
                fp.close()

        if self.embedding_type in ['glove', 'bert', 'skip_gram']:
            clean_data = self.clean_data(data, 'Title')
            with open('output.tsv', 'w', encoding='utf-8') as fp:
                for j, row, in data.iterrows():
                    title = row['Title']
                    clean_title = clean_data['Title'][j]
                    if self.embedding_type == 'glove':
                        tokenized_input = self.vectorizer(clean_title, padding='max_length',
                                                          max_length=90,
                                                          truncation=True,
                                                          return_tensors="np")
                        idxs_title = tokenized_input['input_ids']
                        attention_mask = tokenized_input['attention_mask']
                    else:
                        idxs_title = self.vectorizer(clean_title)
                        idxs_title = np.expand_dims(idxs_title, 0)

                    if self.embedding_type == 'bert':
                        prediction = model.predict([idxs_title, attention_mask])
                    else:
                        prediction = model.predict(idxs_title)

                    predicted_idxs = np.argmax(np.squeeze(prediction), axis=1)
                    Aspect_Name = []
                    Aspect_Value = []
                    for st, idx in zip(title.split(" "), predicted_idxs):
                        if st == 'none':
                            break
                        if self.str_dict_y[idx] not in ['none', '']:
                            Aspect_Name.append(self.str_dict_y[idx])
                            Aspect_Value.append(st)
                        elif self.str_dict_y[idx] == '' and len(Aspect_Value) > 0:
                            print(len(Aspect_Value))
                            Aspect_Value[-1] += ' ' + st
                        else:
                            Aspect_Value.append(st)

                    for i in range(len(Aspect_Name)):
                        fp.write(str(row['Record Number']) + "\t" + Aspect_Name[i] + "\t" + Aspect_Value[i] + '\r')
                fp.close()


