import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
import pdb
import os
import random
import math
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import pickle

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
				 n_classes=2, shuffle=True):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.labels = labels
		self.list_IDs = list_IDs
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y


	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)


	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, *self.dim, self.n_channels))
		y = np.empty((self.batch_size), dtype=int)

		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store sample
			X[i,] = np.load(ID)

			# Store class
			y[i] = self.labels[ID]

		return X, keras.utils.to_categorical(y, num_classes=self.n_classes)






# Parameters
params = {'dim': (32,32),
		  'batch_size': 30,
		  'n_classes': 2,
		  'n_channels': 2,
		  'shuffle': True}

# Datasets
# Get all numpy arrays for talking
t_files   = os.listdir("faces/t/")
random.shuffle(t_files)
nt_files  = os.listdir("faces/nt/")
random.shuffle(nt_files)

# Splitting into training and validation

# Create Talking and No-talking Training files :

t_train_files   = t_files[0:math.floor(0.7*len(t_files))]
t_train_files   = ["faces/t/" + s for s in t_train_files]
nt_train_files  = nt_files[0:math.floor(0.7*len(nt_files))]
nt_train_files  = ["faces/nt/" + s for s in nt_train_files]
train_files     = t_train_files + nt_train_files
train_labels    = [1]*len(t_train_files) + [0]*len(nt_train_files)


# Create Talking and No-talking validation files:

vt_files   = os.listdir("faces_validation/t/")
random.shuffle(vt_files)
vnt_files  = os.listdir("faces_validation/nt/")
random.shuffle(vnt_files)

# Create valid_files and valid_labels ...
vt_train_files   = t_files[0:math.floor(0.3*len(vt_files)):-1]
vt_train_files   = ["faces_validation/t/" + s for s in vt_train_files]

vnt_train_files   = nt_files[0:math.floor(0.3*len(vnt_files)):-1]
vnt_train_files   = ["faces_validation/nt/" + s for s in vnt_train_files]

v_train_files     = vt_train_files + vnt_train_files
v_train_labels    = [1]*len(vt_train_files) + [0]*len(vnt_train_files)


# Creating dictionary for partition
partition = {
	'train':train_files,
	'validation': v_train_files
}

# Create labels dictionary
labels = {}
for idx,cur_label in enumerate(train_labels):
	labels[train_files[idx]] = cur_label

# Create Validation labels here ...
for idx,cur_label in enumerate(v_train_labels):
	labels[v_train_files[idx]] = cur_label

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
model = Sequential()
model = keras.Sequential()

# Modify the input layer for the rectangular size that
# you are given. Also, modify the number of channels.
model.add(keras.layers.Conv2D(filters=6,
                        kernel_size=(3, 3),
                        activation='relu',
                        input_shape=(32,32,2)))
model.add(keras.layers.AveragePooling2D())

model.add(keras.layers.Conv2D(filters=16,
                        kernel_size=(3, 3),
                        activation='relu'))
model.add(keras.layers.AveragePooling2D())

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=120, activation='relu'))
model.add(keras.layers.Dense(units=84, activation='relu'))

print(len(labels))
# Modify the activation and the number of units
# so as to implement a classifier for talking versus no talking.
model.add(keras.layers.Dense(units=2, activation = 'sigmoid'))
model.compile(optimizer='sgd',loss='mse',metrics=['accuracy'])

# Train model on dataset
model_metrics = model.fit_generator(generator=training_generator,
					validation_data=validation_generator,
					use_multiprocessing=False,
					workers=1,
					epochs=100)

print(model_metrics.history['acc'])

#Save the trained model to disk
model.save("modal.h5")
