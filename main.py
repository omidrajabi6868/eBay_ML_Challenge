from tensorflow.keras.models import load_model
from Network import Net
from utils import Helper

import tensorflow as tf
import pickle
import pandas as pd
import os


# Hide GPU from visible devices
# tf.config.set_visible_devices([], 'GPU')


def main():
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         tf.config.experimental.set_virtual_device_configuration(
    #             gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12000)])
    #     except RuntimeError as e:
    #         print(e)

    embedding_types = ['scratch', 'glove', 'skip_gram', 'bert']
    embedding_type = embedding_types[2]

    list_titles_df = pd.read_csv('Data/Listing_Titles.tsv.gz', header=0, delimiter="\t", quoting=3)
    max_spaces = max(title.count(" ") for title in list_titles_df['Title'])
    print('Listing Titles: ')
    print(list_titles_df.head())
    list_titles_df = list_titles_df[:1000000]

    train_tagged_df = pd.read_csv('Data/Train_Tagged_Titles.tsv.gz', header=0, delimiter="\t", quoting=3)
    print('Listing Trained Tags: ')
    print(train_tagged_df.head())

    helper = Helper(list_titles_df, train_tagged_df, max_spaces + 1, embedding_type, embeded_vector_size=50)

    if not os.path.exists('Data/dataset.pkl'):
        helper.build_dataset()

    helper.train_df = pd.read_pickle('Data/dataset.pkl')
    helper.train_df = helper.train_df.fillna("")

    if not os.path.exists('Data/words_list'):
        helper.vocab_builder()

    x_train, y_train = helper.load_data()

    net = Net(vocab_num=len(helper.vocab_to_index),
              embed_vector_size=helper.embeded_vector_size,
              embedding_matrix=helper.embedding_matrix,
              embedding_trainable=False,
              embedding_type=embedding_type)
    net.model.build(input_shape=(helper.seq_len, len(helper.vocab_to_index)))
    net.model.summary()

    if not os.path.exists(f'Models/final_model_{embedding_type}.h5'):
        net.train_model(x_train, y_train, embedding_type)

    net.model.load_weights(f'Models/final_model_{embedding_type}.h5')
    # net.train_model(x_train, y_train, embedding_type)

    quize_data = helper.df_titles[5000:30000]
    helper.test_model(net.model, quize_data)

    return


if __name__ == '__main__':
    main()
