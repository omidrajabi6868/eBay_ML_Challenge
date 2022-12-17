# eBay_ML_Challenge
To run this code you need to just call main.py. However, before doing so, it is better to adjust some parameters based on what result you want to get. 
First, you should determine which embedding methods you like to apply from the below list. 

    embedding_types = ['scratch', 'glove', 'skip_gram', 'bert']
    embedding_type = embedding_types[2]

In Network.py, comment and uncomment the function which will create different models. There are three models: Transformer, bi_directional models, and BERT model. 

        # self.model = self.create_transformer_model(ff_dim=512, num_heads=2)
        self.model = self.create_bidirectional_model()
        # self.model = self.create_bert_model()
