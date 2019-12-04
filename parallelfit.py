import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import pdb
import model.model as model
from time import time
import json

with open("./config.json",'r') as load_f:
  config = json.load(load_f)

data_file = glob.glob("./data/*.txt") + glob.glob("./data/*.csv")

# data_file = glob.glob("./simulate/*.txt") + glob.glob("./simulate/*.csv")

X_data = []
Y_data = []
start = time()
for file_name in data_file:
  print(file_name)
  filepath, tempfilename = os.path.split(file_name)
  filename, extension = os.path.splitext(tempfilename)

  ''' prepare dataset  '''
  data = pd.read_csv(file_name, header=None)
  df = pd.DataFrame(data)
  df.fillna(value=0)
  X = df[0].values
  Y = df[1].values
  
  X = X * config['X_scale']
  X = X.astype(np.float32)
  Y = Y / np.max(Y)
  Y = Y.astype(np.float32)
  X_data.append(X.reshape((len(X), 1)))
  Y_data.append(Y.reshape((len(Y), 1)))

X_data = np.concatenate(X_data, axis=1)
Y_data = np.concatenate(Y_data, axis=1)

''' Build the model '''
particle_num = config['particle_num']
fitmodel = model.Model(particle_num, X_data.shape[1])
  
''' Train the model '''
optimizer = tf.keras.optimizers.Adam(lr=config['learning rate'])
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')

@tf.function
def train_step(X, Y):
  with tf.GradientTape() as tape:
    predictions = fitmodel(X)
    loss = fitmodel.loss(Y, predictions)
  gradients = tape.gradient(loss, fitmodel.trainable_variables)
  optimizer.apply_gradients(zip(gradients, fitmodel.trainable_variables))
  train_loss(loss)
  train_accuracy(Y, predictions)
  return

@tf.function
def test_step(X, Y):
  predictions = fitmodel(X)
  t_loss = fitmodel.accuracy(Y, predictions)
  test_loss(t_loss)
  test_accuracy(Y, predictions)
  return
  
EPOCHS = config['EPOCHS']
steps_per_epoch = config['steps_per_epoch']
for epoch in range(EPOCHS):
  # shuffle
  permutation = np.random.permutation(X_data.shape[0])
  Xi = X_data[permutation]
  Yi = Y_data[permutation]

  # split
  split_num = int(X_data.shape[0] * config['train/test set split ratio'])
  # Train
  X_train = Xi[:split_num]
  Y_train = Yi[:split_num]
  train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).repeat(steps_per_epoch).batch(len(X_train))
  # Test
  X_test = Xi[split_num:]
  Y_test = Yi[split_num:]
  
  # test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).repeat(steps_per_epoch).batch(len(X_test))
  # for x_ds, y_ds in train_ds:
  #   train_step(x_ds, y_ds)
  # for x_ds, y_ds in test_ds:
  #   test_step(x_ds, y_ds)
  
  for step in range(steps_per_epoch):
    train_step(X_train, Y_train)
    test_step(X_test, Y_test)
  
  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result(),
                        test_loss.result(),
                        test_accuracy.result()))
  
  ''' Evaluate accuracy '''
  # Reset the metrics for the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()
  
stop = time()
print("Time consumed :" + str(stop-start) + "s")

Y_fit = fitmodel(X_data)
for ind, file_name in enumerate(data_file):
  filepath, tempfilename = os.path.split(file_name)
  filename, extension = os.path.splitext(tempfilename)
  ''' Saving result '''
  parameters = []
  for i in range(particle_num):
    parameters.append([fitmodel.R0[i].numpy()[ind], 
                      fitmodel.f0[i].numpy()[ind], 
                      fitmodel.dw[i].numpy()[ind], 
                      np.max(df[1].values) * fitmodel.Rm[i].numpy()[ind],
                      np.max(df[1].values) * fitmodel.bg.numpy()[ind]])
  result = pd.DataFrame(parameters, columns=["R0", "f0", "dw", "Rm", "bg"])
  result.to_csv(os.path.join("./result", filename + ".csv"))
  ''' Draw fitting results '''
  plt.scatter(X_data[:, ind], Y_data[:, ind], c='b')
  plt.scatter(X_data[:, ind], Y_fit[:, ind], c='r')
  plt.savefig(os.path.join("./result", filename + ".jpg"))
