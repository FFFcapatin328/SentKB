import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
os.environ['TOKENIZERS_PARALLELISM'] = '0'

# Zero-Shot Stance Detection: A Dataset and Model using Generalized Topic Representations
class VASTZeroFewShot(Dataset):
    def __init__(self, phase, model='bert-base', wiki_model='', senti_flag=False, graph_flag=False, summa_flag=False):
        path = 'data/VAST/'
        graph_feature_base = 'graph_features/'
        senti_feature_base = 'senti_features/'
        if phase in ['train', 'test']:
            file_path = f'{path}/vast_{phase}.csv'
            graph_feature_path = f'{graph_feature_base}/sf_{phase}_300_5000.np'
            senti_path = f'{senti_feature_base}/{phase}_senti.npy'
            summarized_sentences_path = f'{path}/summarized_vast_{phase}.csv'
        else:
            file_path = f'{path}/vast_dev.csv'
            graph_feature_path = f"{graph_feature_base}/sf_dev_300_5000.np"
            senti_path = f'{senti_feature_base}/dev_senti.npy'
            summarized_sentences_path = f'{path}/summarized_vast_{phase}.csv'
        # process dataset
        df = pd.read_csv(file_path)

        print(f'# of {phase} examples: {df.shape[0]}')

        topics = df['topic_str'].tolist()
        tweets = df['text_s'].tolist()
        stances = df['label'].tolist()
        if phase == 'test':
            few_shot = df['seen?'].tolist()
            qte = df['Qte'].tolist()
            sarc = df['Sarc'].tolist()
            imp = df['Imp'].tolist()
            mls = df['mlS'].tolist()
            mlt = df['mlT'].tolist()
        else:
            few_shot = np.zeros(df.shape[0])
            qte = np.zeros(df.shape[0])
            sarc = np.zeros(df.shape[0])
            imp = np.zeros(df.shape[0])
            mls = np.zeros(df.shape[0])
            mlt = np.zeros(df.shape[0])
        
        # os.environ['TRANSFORMERS_OFFLINE'] = '1'
        from transformers import AutoTokenizer
        if model == 'bert-base':
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        elif model == 'bertweet':
            tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
        else: # covid-twitter-bert
            tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')

        if wiki_model:
            wiki_dict = pickle.load(open(f'{path}/wiki_dict.pkl', 'rb'))
            wiki_summaries = df['new_topic'].map(wiki_dict).tolist()

            if wiki_model == model:
                tokenizer_wiki = tokenizer
            else:
                tokenizer_wiki = AutoTokenizer.from_pretrained('bert-base-uncased')

            if wiki_model == model:
                tweets_targets = [f'text: {x} target: {y}' for x, y in zip(tweets, topics)]
                encodings = tokenizer(tweets_targets, wiki_summaries, padding=True, truncation=True)
                encodings_wiki = {'input_ids': [[0]] * df.shape[0], 'attention_mask': [[0]] * df.shape[0]}
            else:
                encodings = tokenizer(tweets, topics, padding=True, truncation=True)
                encodings_wiki = tokenizer_wiki(wiki_summaries, padding=True, truncation=True)

        else:
            print("no wiki")
            encodings = tokenizer(tweets, topics, padding=True, truncation=True)
            encodings_wiki = {'input_ids': [[0]] * df.shape[0], 'attention_mask': [[0]] * df.shape[0]}

        if senti_flag:
            # read summarization
            senti_tokenizer = AutoTokenizer.from_pretrained("twitter-roberta-base-sentiment-latest")
            if summa_flag:
                summa_df = pd.read_csv(summarized_sentences_path) 
                senti_tweets = summa_df['summarized_post'].tolist()
                encodings_senti = senti_tokenizer(senti_tweets, topics, padding=True, truncation=True)
            else:
                print("no summarization")
                encodings_senti = senti_tokenizer(tweets, topics, padding=True, truncation=True)
            input_ids_senti = torch.tensor(encodings_senti['input_ids'], dtype=torch.long)
            attention_mask_senti = torch.tensor(encodings_senti['attention_mask'], dtype=torch.long)
        else:
            print("no senti-bert")
            encodings_senti = {'input_ids': [[0]] * df.shape[0], 'attention_mask': [[0]] * df.shape[0]}
            input_ids_senti = torch.tensor(encodings_senti['input_ids'], dtype=torch.long)
            attention_mask_senti = torch.tensor(encodings_senti['attention_mask'], dtype=torch.long)
        # encodings for the texts and tweets
        input_ids = torch.tensor(encodings['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encodings['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(encodings['token_type_ids'], dtype=torch.long)

        # encodings for wiki summaries
        input_ids_wiki = torch.tensor(encodings_wiki['input_ids'], dtype=torch.long)
        attention_mask_wiki = torch.tensor(encodings_wiki['attention_mask'], dtype=torch.long)

        stances = torch.tensor(stances, dtype=torch.long)
        print(f'max len: {input_ids.shape[1]}, max len wiki: {input_ids_wiki.shape[1]}')

        # process encoded graph features
        if graph_flag:
            graph_features = np.load(graph_feature_path, allow_pickle=True)
            graph_features = torch.tensor(graph_features, dtype=torch.float32)
            print(f'graph features shape: {graph_features.shape}')
        else:
            print("no graph features")
            graph_features = torch.tensor(np.zeros(df.shape[0]), dtype=torch.float32)

        self.phase = phase
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.mlt = mlt
        self.input_ids_wiki = input_ids_wiki
        self.attention_mask_wiki = attention_mask_wiki
        self.input_ids_senti = input_ids_senti
        self.attention_mask_senti = attention_mask_senti
        self.stances = stances
        self.few_shot = few_shot
        self.qte = qte
        self.sarc = sarc
        self.imp = imp
        self.mls = mls
        self.graph_features = graph_features

    def __getitem__(self, index):
        item = {
            'input_ids': self.input_ids[index],
            'attention_mask': self.attention_mask[index],
            'token_type_ids': self.token_type_ids[index],
            'input_ids_wiki': self.input_ids_wiki[index],
            'attention_mask_wiki': self.attention_mask_wiki[index],
            'input_ids_senti':self.input_ids_senti[index],
            'attention_mask_senti':self.attention_mask_senti[index],
            'stances': self.stances[index],
            'few_shot': self.few_shot[index],
            'qte': self.qte[index],
            'sarc': self.sarc[index],
            'imp': self.imp[index],
            'mls': self.mls[index],
            'mlt': self.mlt[index],
            'graph_features': self.graph_features[index],
        }
        return item

    def __len__(self):
        return self.stances.shape[0]


def data_loader(data, phase, topic, batch_size, model='bert-base', wiki_model='', n_workers=4, senti_flag=False, graph_flag=False, summa_flag=False):
    shuffle = True if phase == 'train' else False
    if data == 'vast':
        dataset = VASTZeroFewShot(phase, model=model, wiki_model=wiki_model, senti_flag=senti_flag, graph_flag=graph_flag, summa_flag=summa_flag)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers)
    return loader
