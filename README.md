# The Task of Text Imputation
  A [pytorch](https://pytorch.org/)-implemented neural network to complete sentences with one missing word. Inspired by the concept of Continuous Bags of Words, [CBOW](https://arxiv.org/abs/1301.3781), the model is designed to predict the most likely words when given some context.  
 
# Data Preprocessing
  The author decides to keep all punctuations in training samples, hoping that they will aid the model in better learning syntactical features of the English language. However, while part of the context, punctuations and stop words like prepositions and pronouns are not used as gold labels during training. All words are tokenized and are not phrasified, that is words like "new york" and "san francisco" are considered to be two tokens. The model is trained using a dictionary of 100k words. The last 3 words in the dictionary are reserved for special tokens: <BOS> , <EOS> , and <UNK> denoting the beginning of a sentence , the end of a sentence and an unknown word , respectively. The author finds that using a smaller dictionary results in the model often making unknowns <UNK> predictions. A bigger dictionary on the other hand also adds to training time. Creating a dictionary and data preprocessing are written in preprocess_traindat.py

# Training
  Due to the data's massive size, the network is trained using only parts of the corpus from [1 Billion Word Language Model Benchmark](http://www.statmt.org/lm-benchmark/). The network is based mainly on 2 LSTM neural networks, one taking as input a left context sequence of words and the other taking a right context sequence. For example, given a context sentence < ... w1 w2 w3 ___ w5 w6 w7... >, the left LSTM processes the < .. w1 w2 w3 > sequence while the the right LSTM processes the < ... w7 w6 w5 > sequence to produce a pair of outputs that meets at the missing word's position. The the 2 outputs are then concatenated and mapped to a predicted word w4'. This algorithm can be found in LSTM_bilang_nostop_train.py
  
# Training Sample Generation
  As mentioned above, due to the data's massive size, the author needs an efficient way to read in the sentences from the corpus. The MySentences class in LSTM_bilang_nostop_train.py opens a directory containing multiple files and creates a generator over them. While easy on memory usage, the approach cannot perfom random sampling of sentences since the generator yeilds sequentially. The sentences in corpus files therefore need to be shuffled before the model begins training. Shuffling can be done by calling shuffle_dat.py. With a sentence drawn from the corpus, the program yields multiple training samples in which words (non-punctuations and non-stopwords) are randomly chosen as gold labels. The amount of training samples generated from one sentence is set by default to 80 % of the the sentence's length.
  
# Evaluation
  Because there can be multiple correct missing words for a given context, evaluation of the model is currently based on human judgement. Users have the option of picking the top n most likely missing words during an evaluation. To evaluate the model, run LSTM_bilang_nostop_eval.py. The program will prompt for a context. All contexts should be entered in the following form: 
                                          ... w1 w2 w3 ___ w5 w6 w7 .... 
where 3 underbars indicate the position of a missing word. Punctuations and apostrophes should be separated by space like "don 't". 
  
# Requirements
  1. nltk
  2. pytorch
  3. pickle
  
  
  

  
  
  
