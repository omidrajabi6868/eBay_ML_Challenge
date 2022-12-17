# eBay_ML_Challenge
This code need to have a large dataset from eBay company datasource to be able to run; unfortunately, I coudn't upload it here because of the limited space. But as soon as you found the data, just going through these steps can make it easy to run the data. For the first run, it may take a long time to run, but after buildning the dataset in first run, it shouldn't take long anymore to execute.

To run this code, you need to just run main.py. However, before doing so, it is better to adjust some parameters based on what result you want to get. 
First, you should determine which embedding methods you like to apply from the below list in main.py. 

    embedding_types = ['scratch', 'glove', 'skip_gram', 'bert']
    embedding_type = embedding_types[2]

Second, in Network.py, comment and uncomment the function which will create different models. There are three models: Transformer, bi_directional models, and BERT model. Only one of them must be uncomment. 

        # self.model = self.create_transformer_model(ff_dim=512, num_heads=2)
        self.model = self.create_bidirectional_model()
        # self.model = self.create_bert_model()
        
The metric results such as word by word recall, precision, and f1-score, and also final scores can be found in metrics.txt after running the algorithm. And they will be produced automatically based on which method chosen in previous steps. 

Th output.tsv will produce the test data predictions for the model chosen in previous step. The ground truth for the test data can be found in ground_truth.tsv
