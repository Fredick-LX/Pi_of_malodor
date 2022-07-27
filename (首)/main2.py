with open("data.txt","r",encoding="utf-8") as file:
    content = file.read()
chars = list(set(content))
n_chars = len(chars)
char_indices = dict((c,i) for i,c in enumerate(chars))
indices_chars = dict((i,c) for i,c in enumerate(chars))
maxlen=6
step=1
import numpy as np
from keras.models import load_model
model = load_model("models.LSTM")
def sample(preds):
    preds = np.asarray(preds).astype('float64')
    preds = preds / np.sum(preds)
    return np.argmax(preds)
def creater(inputs,length):
    words = ""
    for i in range(length):
        x_preds = np.zeros((1,maxlen,n_chars))
        for i,chars in enumerate(inputs):
            x_preds[0,i,char_indices[chars]]=1
        preds = model.predict(x_preds,verbose=0)[0]
        next_index = sample(preds)
        next_char = indices_chars[next_index]
        words += next_char
        inputs = inputs[1:] + next_char
    return words
def main(inputs,times):
    output=inputs
    print("input:",output)
    for i in range(times):
        last=inputs
        new=creater(inputs,1)
        inputs=last[1:]+new
        output+=new
    print("output:",output)
while True:
    print("\n"*5)
    print("----input----")
    start=input("input:")
    long=int(input("length:"))
    print("----output----")
    main(start,long)