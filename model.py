import tensorflow as tf
import numpy as np
import pdb
class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    ''' initialize_all_variables'''
    self.R0 = tf.Variable(np.float32(np.random.uniform(0.5, 0.97, 1)))
    self.f0 = tf.Variable(np.float32(np.random.uniform(2850, 2890, 1)))
    self.dw = tf.Variable(np.float32(np.random.uniform(2, 20, 1)))
    self.Rm = tf.Variable(np.float32(1.0))
    self.bg = tf.Variable(np.float32(0))
  
  def call(self, f):
    f0 = self.f0
    R0 = self.R0
    dw = self.dw
    Rm = self.Rm
    bg = self.bg
    y = Rm * (4 * ((f - f0) ** 2) + (dw ** 2) * R0) / (4 * ((f - f0) ** 2) + (dw ** 2)) + bg
    return y
  
  @staticmethod
  def loss(predicted_y, true_y):
    return tf.reduce_mean(tf.square(predicted_y - true_y))

  @staticmethod
  def accuracy(predicted_y, true_y):
    return tf.reduce_mean(tf.square(predicted_y - true_y))