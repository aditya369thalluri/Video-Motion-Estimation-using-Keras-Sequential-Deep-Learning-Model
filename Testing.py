import cv2
import pickle
import numpy as np
import keras
from keras.models import Sequential
import pdb
import os

 #Testing

#Load the Trained Model from disk

model = load('model.h5')

print('\n Testing Face 2 from Testing video')
xtest = nt_files[1]
xtest = np.load("faces/nt/"+xtest)
xtest = np.expand_dims(xtest,axis=0)

#xtest = scalar.fit_transform(xtest)



result = model.predict(xtest)
print(result)
ypred = []
if result[0][0]>0.5:
    print('Not talking')
    ypred.append(0)
else:
    print('Talking')
    ypred.append(1)


print(ypred)
