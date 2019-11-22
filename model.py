import tensorflow as tf
import numpy as np
import pdb

class Model(tf.keras.Model):
  def __init__(self, particle_num):
    super(tf.keras.Model, self).__init__()
    ''' initialize_all_variables'''
    self.particle_num = particle_num
    self.R0 = []
    self.f0 = []
    self.dw = []
    self.Rm = []
    for _ in range(particle_num):
      self.R0.append(tf.Variable(np.float32(np.random.uniform(0.5, 0.97, 1))))
      self.f0.append(tf.Variable(np.float32(np.random.uniform(2850, 2890, 1))))
      self.dw.append(tf.Variable(np.float32(np.random.uniform(2, 20, 1))))
      self.Rm.append(tf.Variable(np.float32(1.0)))
    # self.bg = tf.Variable(np.float32(0))
  
  def call(self, f):
    f0 = self.f0
    R0 = self.R0
    dw = self.dw
    Rm = self.Rm
    # bg = self.bg
    particle_list = [Rm[i] * (4 * ((f - f0[i]) ** 2) + (dw[i] ** 2) * R0[i]) / (4 * ((f - f0[i]) ** 2) + (dw[i] ** 2)) for i in range(self.particle_num)]
    y = tf.math.add_n(particle_list)
    return y
  
  @staticmethod
  def loss(predicted_y, true_y):
    return tf.reduce_mean(tf.square(predicted_y - true_y))

  @staticmethod
  def accuracy(predicted_y, true_y):
    return tf.reduce_mean(tf.square(predicted_y - true_y))

