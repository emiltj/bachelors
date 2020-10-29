import sklearn as sk
import matplotlib as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

x=3
x
test = pd.read_csv('./csv_files/test_selected_features.csv')
train = pd.read_csv('./csv_files/train_selected_features.csv')

train = train.drop('Unnamed: 0', axis=1)
test = test.drop('Unnamed: 0', axis=1)

train.head()

import sys
print(sys.version)

