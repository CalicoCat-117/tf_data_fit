import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import pdb
import model

def fake_data_generator(xs, model, var_scale, point_num):
  cites = np.random.randint(0, len(xs) - 1, size=point_num)
  X = xs[cites]
  rand_noise = np.random.normal(loc=0.0, scale=var_scale, size=point_num)
  X = X.astype(np.float32)
  Y = model(X).numpy() + rand_noise
  return X, Y

def save_csv():
  return


if __name__ == '__main__':
  xs = np.linspace(2800, 3200, 500)
  particle_num = 3
  build_model = model.Model(particle_num)
  build_model.f0 = [tf.Variable(np.float32(2871)), 
                    tf.Variable(np.float32(3001)), 
                    tf.Variable(np.float32(3050))]
  var_scale = 0.03
  point_num = 10000
  X, Y = fake_data_generator(xs, build_model, var_scale, point_num)
  X = X / 1000
  Y = Y * 1000
  X = X.reshape((point_num,1))
  Y = Y.reshape((point_num,1))
  data = np.concatenate((X, Y),axis=1)
  df = pd.DataFrame(data)
  df.to_csv(os.path.join("./simulate", "sim1" + ".txt"), index=False, header=False)
  plt.scatter(X, Y, c='b')
  plt.show()