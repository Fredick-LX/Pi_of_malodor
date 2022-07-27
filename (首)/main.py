with open("data.txt","r",encoding="utf-8") as file:
    content = file.read()
chars = list(set(content))
n_chars = len(chars)
char_indices = dict((c,i) for i,c in enumerate(chars))
indices_chars = dict((c,i) for i,c in enumerate(chars))
maxlen=6
step=1
sentences = []
next_chars = []
for i in range(0,len(content)-maxlen,step):
    sentences.append(content[i:i+maxlen])
    next_chars.append(content[i+maxlen])
import numpy as np
try:
    X=np.load("X_data.npy")
    Y=np.load("Y_data.npy")
except:
    X = np.zeros((len(sentences),maxlen,len(chars)),dtype=bool)
    Y = np.zeros((len(sentences),len(chars)),dtype=bool)
    for i,sentence in enumerate(sentences):
        for j,char in enumerate(sentence):
            X[i,j,char_indices[char]]=1
        Y[i,char_indices[next_chars[i]]]=1
    np.save("X_data.npy",X)
    np.save("Y_data.npy",Y)
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.optimizers import rmsprop_v2
model = Sequential()
model.add(LSTM(units=50,input_shape=(maxlen,n_chars)))
model.add(Dense(units=n_chars,activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer=rmsprop_v2.RMSprop(learning_rate=0.03))
model.summary()
model.fit(X,Y,batch_size=256,epochs=50)
model.save("models.LSTM")
print()
print("The model has been generated")