import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import pdb
import model


data_file = glob.glob("./data/*.txt") + glob.glob("./data/*.csv")



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
  
  X = X * 1000
  X = X.astype(np.float32)
  Y = Y / np.max(Y)
  Y = Y.astype(np.float32)
  ''' Build the model '''
  fitmodel = model.Model()
  
  ''' Train the model '''
  optimizer = tf.keras.optimizers.Adam(lr=0.01)
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
  
  EPOCHS = 5000
  for epoch in range(EPOCHS):
    # shuffle
    permutation = np.random.permutation(len(X))
    Xi = X[permutation]
    Yi = Y[permutation]
    # split
    split_num = int(len(X) * 0.9)
    # Train
    X_train = Xi[:split_num]
    Y_train = Yi[:split_num]
    # Test
    X_test = Xi[split_num:]
    Y_test = Yi[split_num:]
    
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
  
  ''' Saving result '''
  result = pd.DataFrame([fitmodel.R0.numpy()[0], fitmodel.f0.numpy(), fitmodel.dw.numpy()[0], fitmodel.Rm.numpy(), fitmodel.bg.numpy()])
  result.to_csv(os.path.join("./result", filename + ".csv"))
  ''' Draw fitting results '''
  plt.scatter(X, Y, c='b')
  plt.scatter(X, fitmodel(X), c='r')
  plt.show()
