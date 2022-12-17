from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding, GRU, Dropout
from transformers import TFAutoModel
from utils import positional_encoding

import tensorflow as tf
import datetime
import numpy as np


class Net:

    def __init__(self, vocab_num, embed_vector_size=32,
                 embedding_matrix='uniform', embedding_trainable=False, embedding_type='scratch'):
        self.learning_rate = 1e-3
        self.epochs = 500
        self.vocab_num = vocab_num
        self.embedding_trainable = embedding_trainable
        self.embed_vector_size = embed_vector_size
        if type(embedding_matrix) != type('uniform'):
            self.embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix)
        else:
            self.embeddings_initializer = 'uniform'
        # self.model = self.create_transformer_model(ff_dim=512, num_heads=2)
        self.model = self.create_bidirectional_model()
        # self.model = self.create_bert_model()

    def create_bidirectional_model(self):
        model = tf.keras.Sequential()
        model.add(Embedding(self.vocab_num,
                            self.embed_vector_size,
                            embeddings_initializer=self.embeddings_initializer,
                            trainable=self.embedding_trainable,
                            input_length=59))

        model.add(Bidirectional(LSTM(512, return_sequences=True, activation='relu'), input_shape=(59, self.embed_vector_size)))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(34, activation='softmax'))

        opt = tf.keras.optimizers.Adam(self.learning_rate)
        loss = CustomNonPaddingTokenLoss()
        model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

        return model

    def create_transformer_model(self, ff_dim, num_heads):
        model = TransformerModel(34, vocab_size=self.vocab_num,
                                 embedding_matrix=self.embeddings_initializer,
                                 embedding_trainable=self.embedding_trainable,
                                 embed_dim=self.embed_vector_size,
                                 num_heads=num_heads, ff_dim=ff_dim, maxlen=59)

        opt = tf.keras.optimizers.Adam(self.learning_rate)
        loss = CustomNonPaddingTokenLoss()
        model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

        return model

    def create_bert_model(self):

        base_model = TFAutoModel.from_pretrained("bert-base-uncased", trainable=True)
        input_ids = tf.keras.layers.Input(shape=(90,), name="input_ids", dtype="int32")
        attenion_mask = tf.keras.layers.Input(shape=(90,), name="attention_mask", dtype="int32")
        x = base_model(input_ids, attention_mask=attenion_mask)

        x = Bidirectional(LSTM(512, return_sequences=True, activation='relu'))(x.last_hidden_state)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(34, activation='softmax')(x)

        opt = tf.keras.optimizers.Adam(self.learning_rate)
        loss = CustomNonPaddingTokenLoss()

        model = tf.keras.Model([input_ids, attenion_mask], x)
        model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

        return model

    def train_model(self, x, y, embedding_type):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        path_checkpoint = f"Models/final_model_{embedding_type}.h5"
        es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=50)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.0001)

        modelckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            monitor="val_loss",
            filepath=path_checkpoint,
            verbose=1,
            save_best_only=True,
            save_weights_only=True)

        self.model.fit(x, y,
                       batch_size=64,
                       validation_split=0.05,
                       epochs=self.epochs,
                       callbacks=[modelckpt_callback, es_callback, tensorboard_callback, reduce_lr])

        return self.model


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, embedding_matrix, embedding_trainable):
        super(TokenAndPositionEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim,
                                                   embeddings_initializer=embedding_matrix,
                                                   trainable=embedding_trainable,
                                                   input_length=maxlen)
        self.pos_encoding = positional_encoding(length=vocab_size, depth=embed_dim)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        x = self.embedding(inputs)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class TransformerModel(tf.keras.Model):
    def __init__(
            self, num_tags, vocab_size, embedding_matrix, embedding_trainable, maxlen=59,
            embed_dim=32, num_heads=2, ff_dim=32,
    ):
        super(TransformerModel, self).__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim,
                                                         embedding_matrix, embedding_trainable)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.ff = tf.keras.layers.Dense(ff_dim, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.ff_final = tf.keras.layers.Dense(num_tags, activation="softmax")

    def call(self, inputs, training=False):
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = self.dropout1(x, training=training)
        x = self.ff(x)
        x = self.dropout2(x, training=training)
        x = self.ff_final(x)
        return x


class CustomNonPaddingTokenLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_ner_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE
        )
        loss = loss_fn(y_true, y_pred)
        mask = tf.cast((y_true[:, :, 33] != 1), dtype=tf.float32)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)
