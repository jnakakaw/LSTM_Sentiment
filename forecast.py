import numpy as np
import random
import os

import torch
torch.use_deterministic_algorithms(True)  # to help make code deterministic
torch.backends.cudnn.benchmark = False  # to help make code deterministic
import torch.nn as nn

np.random.seed(0)  # to help make code deterministic
torch.manual_seed(0)  # to help make code deterministic
random.seed(0)  # to help make code deterministic

import pickle

from UDA_pytorch_utils import UDA_pytorch_classifier_fit, \
    UDA_pytorch_classifier_predict, \
    UDA_compute_accuracy, UDA_get_rnn_last_time_step_outputs

def sentiment(text):
    if text == None:
        return ''
    else:
        with open('./lstm/vocab.pkl', 'rb') as input:
            vocab = pickle.load(input)

        with open('./lstm/embedding_matrix.pkl', 'rb') as input:
            embedding_matrix = pickle.load(input)

        with open('./lstm/tokenizer.pkl', 'rb') as input:
            tokenizer = pickle.load(input)

        def tokenizer_lowercase(text):
            return [token.lower() for token in tokenizer(text)]
            
        simple_lstm_model = torch.load("./lstm/sentiment_lstm.pt")

        sent_val = UDA_pytorch_classifier_predict(simple_lstm_model,
                                [vocab(tokenizer_lowercase(text))],
                                rnn=True).numpy()[0]

        if sent_val == 1:
            output = 'Your review is positive :)'
        else:
            output = 'Your review is negative :('
        
        return output


