#%%
import pandas as pd 
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import gensim
from tensorflow.keras.layers import Dense,Embedding,LSTM,Dropout
from tensorflow.keras.models import Sequential,load_model
from sklearn.model_selection import train_test_split
import string
# %%
dataset=pd.read_csv("dataset.csv")
# %%
dataset.head()
# %%
def optimizasyon(dataset):
    dataset=dataset.dropna()
    stop_words =set(stopwords.words("turkish"))
    noktamaIsaretleri = set(string.punctuation)
    stop_words.update(noktamaIsaretleri)
    for ind in dataset.index:
        body=dataset["Body"][ind]
        body=body.lower()
        body=re.sub(r'http\S+','',body)
        body=(" ").join([word for word in body.split() if not word in stop_words])
        body="".join([char for char in body if not char in noktamaIsaretleri])
        dataset['Body'][ind]=body
    return dataset    
dataset=optimizasyon(dataset)

# %%
def trASCICevirici(metin):
    translationTable =str.maketrans("ğĞıİöÖüÜşŞçÇ","gGiIoOuUsScC")
    metin =metin.translate(translationTable)
    return metin
# %%
X=dataset.loc[:,"Body"]
y=dataset.loc[:,"Label"]
# %%
X_egitim,X_test,y_egitim,y_test=train_test_split(X,y,test_size=0.2,random_state=28)
X_egitim_dizi=[metin.split() for metin in X_egitim]

maxmesafe=2
minfrekans=1
vektor_boyut=200
w2v_model =gensim.models.Word2Vec(sentences=X_egitim_dizi,vector_size=vektor_boyut,window=maxmesafe,min_count=minfrekans)

# %%
tokenizer=Tokenizer()
tokenizer.fit_on_texts(X_egitim_dizi)
X_egitim_tok=tokenizer.texts_to_sequences(X_egitim_dizi)
kelime_index=tokenizer.word_index
maxlen=100
X_egitim_tok_pad=pad_sequences(X_egitim_tok,maxlen=maxlen)

kelime_sayi=len(kelime_index)+1
print("Sözlük boyutu:",kelime_sayi)

matris=np.zeros((kelime_sayi,vektor_boyut))
for kelime, i in kelime_index.items():
    matris[i]=w2v_model.wv[kelime]
model=Sequential()
model.add(Embedding(matris.shape[0],
                    output_dim=matris.shape[1],
                    weights=[matris],
                    input_length=maxlen,
                    trainable=False))
model.add(LSTM(units=32))
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["acc"])
model.summary()

model.fit(X_egitim_tok_pad,y_egitim,validation_split=0.2,epochs=30,batch_size=32,verbose=1)
model.save("egitilmis_model2.h5")
print("Model eğitildi ve kaydedildi.    ")