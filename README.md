# text_imputation
An LSTM implemented with pytorch to complete sentences with a missing word. Inspired by the concept of Continuous Bags of Words, CBOW, the model predicts the most likely words when given some context.  

# Training
  Due to the data's massive size, the model is trained using only parts of the corpus from the 1 Billion Word Language Model Benchmark. The model is based mainly on 2 LSTM neural networks, each taking in a lateral sequence of words from a given context sentence. For instance, given a context sentence < ... w1 w2 w3 ___ w5 w6 w7... >, the left LSTM processes the < .. w1 w2 w3 > sequence while the the right LSTM processes the < ... w7 w6 w5 > sequence to produce a pair of outputs that meets at the missing word's position. The model then simply concatenates the 2 outputs and maps the concatenation to a predicted word w4. This algorithm can be found in LSTM_bilang_nostop_train.py
  
 # Data Processing
  The author decides to keep all punctuations in training samples, hoping that they will aid the model in learning better syntactical features of the English language. 
  
