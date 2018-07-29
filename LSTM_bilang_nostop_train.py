import random
import string
import os
import io
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nltk
from nltk.corpus import stopwords


# a class to read in multiple training files contained in one dir
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
    # a generator over text file
    def yielddat(self):
        for fname in os.listdir(self.dirname):
            for line in io.open(os.path.join(self.dirname, fname), encoding="ISO-8859-1"):
                sentence = nltk.word_tokenize(line.lower())
                #sample missing word (break point)
                #the sample size is 80% of sentence's length
                sent_length = len(sentence)
                #filter out sentences longer than 20 tokens and shorter than 8 tokens
                if(sent_length > 20 or sent_length < 8):
                    continue
                for _ in range(int(sent_length*(8.0/10.0))):
                    blkpos = random.randint(0,len(sentence)-1)
                    label = sentence[blkpos].encode('utf-8')
                    #discard label if it's punctuation
                    if (label in string.punctuation or label in set(stopwords.words('english')) or label == '``'):
                        continue
                    #create left context
                    left_context  = [ x.encode('utf-8') for x in sentence[0:blkpos]]
                    left_context.insert(0,"<BOS>") #prepend <BOS> to left context
                    right_context = [ y.encode('utf-8') for y in sentence[blkpos+1:sent_length]]
                    right_context.insert(len(right_context), "<EOS>") #append <EOS> to right context

                    #generator yields ([label], [l1, l2, l3..], [... r3, r2, r1])
                    #note that right context is reversed
                    yield [label], left_context, list(reversed(right_context))

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


####################################################################################

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
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size) #x2 because we concatenate left and right output
        self.hidden_left = self.init_hidden_left()
        self.hidden_right = self.init_hidden_right()

    def init_hidden_left(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, 1, self.hidden_dim),
                torch.zeros(2, 1, self.hidden_dim)) # 2 hidden layers

    def init_hidden_right(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, 1, self.hidden_dim),
                torch.zeros(2, 1, self.hidden_dim)) # 2 hidden layers

    def forward(self, left_context, right_context):
        #get embeddings for left and right contexts
        left_embeds = self.word_embeddings(left_context)
        right_embeds = self.word_embeddings(right_context)

        #compute the hidden outputfor left and right contexts
        lstm_out_left, self.hidden_left = self.lstm_left(
            left_embeds.view(len(left_context), 1, -1), self.hidden_left)
        lstm_out_right, self.hidden_right = self.lstm_right(
            right_embeds.view(len(right_context), 1, -1), self.hidden_right)

        #concatenate the left and right output hidden layer(at the last time step, where the missing word is)
        left_right = torch.cat((lstm_out_left[len(left_context)-1], lstm_out_right[len(right_context)-1]))
        #left_right = (lstm_out_left[len(left_context)-1] + lstm_out_right[len(right_context)-1])/2
        #map concatenated hidden output to tag vector
        tag_space = self.hidden2tag(left_right.view( -1, self.hidden_dim*2))
        #compute scores for each tag
        tag_space = F.relu(tag_space)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

#######################################
############ Training Setup ###########
#######################################
#MySentences class takes in a DIRECTORY containing files
train_dir = raw_input("Enter corpus directory: ")
saved_name = raw_input("Save model as: ")
Sentences = MySentences(train_dir).yielddat()
#load 100k word dictionary
print("Start loading dictionary...")
dict = load_obj("lm_dict_all_100000")
print("Finished loading dictionary.")

EMBEDDING_DIM = 100
HIDDEN_DIM = 100

#create the model
model = LSTM_LM(EMBEDDING_DIM, HIDDEN_DIM, len(dict), len(dict))
loss_function = nn.NLLLoss() #define loss
optimizer = optim.SGD(model.parameters(), lr=0.1) #, lr=0.1


#######################################
############ Training #################
#######################################
for epoch in range(1):
    print("START Training with epoch" + str(epoch))
    total_loss = 0
    for s in range(1000000):# input 1 million samples
        label, left_cont, right_cont = next(Sentences)
        # Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden_left = model.init_hidden_left()
        model.hidden_right = model.init_hidden_right()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        left = prepare_sequence(left_cont, dict)
        right = prepare_sequence(right_cont, dict)
        target = prepare_sequence(label, dict)

        # Step 3. Run our forward pass.
        tag_scores = model(left, right)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss = loss_function(tag_scores, target)
        total_loss = total_loss +loss.item()
        if( s % 20 == 0):
            print(s, total_loss/20)
            total_loss = 0
        # print(s, loss.item())
        loss.backward()
        optimizer.step()
        #save model every 10000 samples
        if( s % 10000 == 0):
            torch.save(model, "models/" + saved_name)

torch.save(model, "models/" + saved_name)
