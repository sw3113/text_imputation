import torch
import nltk
import os
import io
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import operator


#map token array to index array in preparation for word embedding layer
def prepare_sequence(seq, to_ix):
    indxs = []
    for w in seq:
        if w not in to_ix:
            indxs.append(to_ix["<UNK>"])
        else:
            indxs.append(to_ix[w])
    return torch.LongTensor(indxs)
#for loading a dict
def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


#######################################
############ Model Definition #########
#######################################
class LSTM_LM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTM_LM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm_left = nn.LSTM(embedding_dim, hidden_dim, num_layers = 2)
        self.lstm_right = nn.LSTM(embedding_dim, hidden_dim, num_layers =2)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size) #x2 because we concatenate left and right output
        self.hidden_left = self.init_hidden_left()
        self.hidden_right = self.init_hidden_right()

    def init_hidden_left(self):
        return (torch.zeros(2, 1, self.hidden_dim),
                torch.zeros(2, 1, self.hidden_dim)) # 2 hidden layers

    def init_hidden_right(self):
        return (torch.zeros(2, 1, self.hidden_dim),
                torch.zeros(2, 1, self.hidden_dim)) # 2 hidden layers

    def forward(self, left_context, right_context):
        #get embeddings for left and right contexts
        left_embeds = self.word_embeddings(left_context)
        right_embeds = self.word_embeddings(right_context)

        #compute the hidden outputfor left and rightcontexts
        lstm_out_left, self.hidden_left = self.lstm_left(
            left_embeds.view(len(left_context), 1, -1), self.hidden_left)
        lstm_out_right, self.hidden_right = self.lstm_right(
            right_embeds.view(len(right_context), 1, -1), self.hidden_right)

        #concatenate the left and right output hidden layer(at the last time step)
        left_right = torch.cat((lstm_out_left[len(left_context)-1], lstm_out_right[len(right_context)-1]))
        #map concatenated hidden output to tag vector
        tag_space = self.hidden2tag(left_right.view( -1, self.hidden_dim*2))
        #compute scores for each tag
        tag_space = F.relu(tag_space)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

#######################################
############ Load Saved Model #########
#######################################
model_path = raw_input("Enter path to model: ")
print("start loading model...")
model = torch.load(model_path)
print("finish loading model...")
print("start loading dictionary...")
dict = load_obj("lm_dict_all_100000")
index2word = {v: k for k, v in dict.iteritems()}
print("finish loading dictionary...")
cand_num = raw_input("Enter number of top candidates: ")
cand_num = int(cand_num)

while(True):
    with torch.no_grad():
        input = raw_input("Enter context < w1 w2 ___ w3 w4 w5 > : ")
        if(input == "quit"):
            break
        sent = nltk.word_tokenize(input.lower())
        blankind = 0
        for i, w in enumerate(sent):
            if(w == "___"):
                blankind = i

        left_context  = sent[0: blankind]
        left_context.insert(0,"<BOS>") #prepend <BOS> to left context
        right_context = sent[blankind+1: len(sent)]
        right_context.insert(len(right_context), "<EOS>") #append <EOS> to right context

        left = prepare_sequence(left_context, dict)
        right = prepare_sequence(list(reversed(right_context)), dict)

        tag_scores = model(left, right)
        #sort candidate missing words
        sorted_list = sorted(enumerate(tag_scores[0]), key=operator.itemgetter(1), reverse = True)
        for index , value in sorted_list[0:cand_num]:
            print(index, index2word[index])
