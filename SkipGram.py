from tensorflow.keras import layers

import io
import pickle
import re
import string
import tqdm
import numpy as np
import tensorflow as tf
import os

seed = 42
AUTOTUNE = tf.data.AUTOTUNE


class skip_gram:

    def __init__(self, sequences, vocab_size, vocab_to_idx, embedding_dim, window_size, num_ns):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.vocab_to_index = vocab_to_idx
        if not os.path.exists('Data/word2vec_dataset'):
            self.dataset = self.generate_training_data(sequences, window_size, num_ns)
        else:
            with open("Data/word2vec_dataset" + ".pickle", 'rb') as pickle_file:
                element_spec = pickle.load(pickle_file)
            self.dataset = tf.data.experimental.load("Data/word2vec_dataset", element_spec=element_spec)
        self.word2vec = Word2Vec(self.vocab_size, self.embedding_dim, num_ns)

    def generate_training_data(self, sequences, window_size, num_ns):

        targets, contexts, labels = [], [], []

        sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(self.vocab_size)

        # Iterate over all sequences (sentences) in the dataset.
        for sequence in tqdm.tqdm(sequences):

            # Generate positive skip-gram pairs for a sequence (sentence).
            positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
                sequence,
                vocabulary_size=self.vocab_size,
                sampling_table=sampling_table,
                window_size=window_size,
                negative_samples=0)

            # Iterate over each positive skip-gram pair to produce training examples
            # with a positive context word and negative samples.
            for target_word, context_word in positive_skip_grams:
                context_class = tf.expand_dims(
                    tf.constant([context_word], dtype="int64"), 1)
                negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                    true_classes=context_class,
                    num_true=1,
                    num_sampled=num_ns,
                    unique=True,
                    range_max=self.vocab_size,
                    seed=seed,
                    name="negative_sampling")

                # Build context and label vectors (for one target word)
                context = tf.concat([tf.squeeze(context_class, 1), negative_sampling_candidates], 0)
                label = tf.constant([1] + [0] * num_ns, dtype="int64")

                # Append each element from the training example to global lists.
                targets.append(target_word)
                contexts.append(context)
                labels.append(label)

        BATCH_SIZE = 1024
        BUFFER_SIZE = 100000
        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
        tf.data.experimental.save(dataset=dataset, path="Data/word2vec_dataset")
        with open("Data/word2vec_dataset" + ".pickle", 'wb') as file:
            pickle.dump(dataset.element_spec, file)

        return dataset

    def word2vec_training(self, lr=1e-5, epochs=0):
        opt = tf.keras.optimizers.Adam(lr)
        self.word2vec.compile(optimizer=opt,
                         loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
        self.word2vec.fit(self.dataset, batch_size=128, epochs=epochs, callbacks=[tensorboard_callback])
        self.word2vec.save('Models/word2vec_model.tf')

        return self.word2vec


class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(Word2Vec, self).__init__()
        self.target_embedding = layers.Embedding(vocab_size,
                                                 embedding_dim,
                                                 input_length=1,
                                                 name="w2v_embedding")
        self.context_embedding = layers.Embedding(vocab_size,
                                                  embedding_dim,
                                                  input_length=num_ns + 1)

    def custom_loss(self, x_logit, y_true):
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)

    def call(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        # target: (batch,)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        # context_emb: (batch, context, embed)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        # dots: (batch, context)
        return dots
