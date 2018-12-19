import os
import pickle
import numpy as np
from keras.models import Sequential
import gensim
from keras.layers.recurrent import LSTM,SimpleRNN
from sklearn.model_selection import train_test_split
import theano
theano.config.optimizer="None"

with open('conversation.pickle') as f:
    vec_x,vec_y=pickle.load(f)

#--------------------------------------------------------------
vec_x=np.array(vec_x)

myarr_x = np.zeros((86,15,300),dtype=float)
x = 0
y = 0
z = 0
for ans in vec_x:
	y=0
	for each_ans in ans:
		for jj in each_ans:
			z=0
			myarr_x[x][y][z]=jj
			z=z+1
		y=y+1
	x=x+1

myarr_x=np.array(myarr_x)

#--------------------------------------------------------------
vec_y=np.array(vec_y)

myarr_y = np.zeros((86,15,300),dtype=float)
x = 0
y = 0
z = 0
for ans in vec_y:
	y=0
	for each_ans in ans:
		z=0
		for jj in each_ans:
			myarr_y[x][y][z]=jj
			z=z+1
		y=y+1
	x=x+1

myarr_y=np.array(myarr_y)

#=--------------------------------------------------------------

vec_x=myarr_x
vec_y=myarr_y

x_train,x_test, y_train,y_test = train_test_split(vec_x, vec_y, test_size=0.2, random_state=1)

model=Sequential()
model.add(LSTM(output_dim=300,input_shape=(15,300),return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300,input_shape=(15,300),return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300,input_shape=(15,300),return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(output_dim=300,input_shape=(15,300),return_sequences=True, init='glorot_normal', inner_init='glorot_normal', activation='sigmoid'))
model.compile(loss='cosine_proximity', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
model.save('LSTM500.h5');
model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
model.save('LSTM1000.h5');
model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
model.save('LSTM1500.h5');
model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
model.save('LSTM2000.h5');
model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
model.save('LSTM2500.h5');
model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
model.save('LSTM3000.h5');
model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
model.save('LSTM3500.h5');
model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
model.save('LSTM4000.h5');
model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
model.save('LSTM4500.h5');
model.fit(x_train, y_train, nb_epoch=500,validation_data=(x_test, y_test))
model.save('LSTM5000.h5');
predictions=model.predict(x_test)
mod = gensim.models.Word2Vec.load('word2vec.bin');

[mod.most_similar([predictions[10][i]])[0] for i in range(15)]
