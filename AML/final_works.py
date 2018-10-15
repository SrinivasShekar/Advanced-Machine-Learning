import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import itertools
import re
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import keras.layers as lyr
from keras.models import Model
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer

##LOADING THE DATASET
df_train = pd.read_csv('train.csv', encoding='utf-8')
df_train = df_train.dropna()
df_train['id'] = df_train['id'].apply(str)


df_all = df_train
df_all['question1'].fillna('', inplace=True)
df_all['question2'].fillna('', inplace=True)

counts_vectorizer = CountVectorizer(max_features=10000-1).fit(
    itertools.chain(df_all['question1'], df_all['question2']))
other_index = len(counts_vectorizer.vocabulary_)

words_tokenizer = re.compile(counts_vectorizer.token_pattern)

##PADDING THE SEQUENCES
def create_padded_seqs(texts, max_len=10):
    seqs = texts.apply(lambda s: 
        [counts_vectorizer.vocabulary_[w] if w in counts_vectorizer.vocabulary_ else other_index
         for w in words_tokenizer.findall(s.lower())])
    return pad_sequences(seqs, maxlen=max_len)

X1_train, X1_val, X2_train, X2_val, y_train, y_val = \
    train_test_split(create_padded_seqs(df_all[df_all['id'].notnull()]['question1']), 
                     create_padded_seqs(df_all[df_all['id'].notnull()]['question2']),
                     df_all[df_all['id'].notnull()]['is_duplicate'].values,
                     stratify=df_all[df_all['id'].notnull()]['is_duplicate'].values,
                     test_size=0.15, random_state=1997)

input1_tensor = lyr.Input(X1_train.shape[1:])
input2_tensor = lyr.Input(X2_train.shape[1:])


input_list = list(df_all['question1'])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_list)
vocab = tokenizer.word_index
vocab_counts = tokenizer.word_counts
sequences = tokenizer.texts_to_sequences(input_list)

embeddings_index = {}
f = open('glove.6B.100d.txt',encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


embedding_matrix = np.zeros((len(vocab) + 1,100))
for word, i in vocab.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer1 = Embedding(len(vocab) + 1,100,weights=[embedding_matrix],input_length=1000,trainable=False)

input_list2 = list(df_all['question2'])
tokenizer2 = Tokenizer()
tokenizer2.fit_on_texts(input_list2)
vocab2 = tokenizer2.word_index
vocab_counts2 = tokenizer2.word_counts
sequences2 = tokenizer2.texts_to_sequences(input_list2)

embedding_matrix2 = np.zeros((len(vocab2) + 1,100))
for word, i in vocab2.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix2[i] = embedding_vector

embedding_layer2 = Embedding(len(vocab2) + 1,100,weights=[embedding_matrix2],input_length=1000,trainable=False)

seq_embedding_layer = lyr.GRU(256, activation='tanh')
merge_layer = lyr.multiply([seq_embedding_layer(embedding_layer1(input1_tensor)), seq_embedding_layer(embedding_layer2(input2_tensor))])

ouput_layer = lyr.Dense(1, activation='sigmoid')(merge_layer)

model = Model([input1_tensor, input2_tensor], ouput_layer)

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])

print(model.summary())

model.fit([X1_train, X2_train], y_train, 
          validation_data=([X1_val, X2_val], y_val), 
          batch_size=128, epochs=6, verbose=2)

features_model = Model([input1_tensor, input2_tensor], merge_layer)
features_model.compile(loss='mse', optimizer='adam',metrics=['acc'])

F_train = features_model.predict([X1_train, X2_train], batch_size=128)
F_val = features_model.predict([X1_val, X2_val], batch_size=128)
print(F_val)