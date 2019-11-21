# import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import glob
import csv
import pandas as pd

data_file = glob.glob("./data/*.txt") + glob.glob("./data/*.csv")
for file_name in data_file:
  print(file_name)
  csvFile = open(file_name, "r")
  reader = csv.reader(csvFile)
  
