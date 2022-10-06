from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

np.random.seed(3)
tf.random.set_seed(3)
df = pd.read_csv('C:\\2022pythonProject\\iris.csv', names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])