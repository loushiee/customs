import math
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import os
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)


class CustomsDataModel:
   def __init__(self, pickle_file, output_folder):
      if not os.path.exists(pickle_file):
         print(f'File not found: {pickle_file}')
         exit()

      # Create the output folder
      if not os.path.exists(output_folder):
         os.mkdir(output_folder)

      self.pickle_file = pickle_file  # Dataset to read and visualize
      self.df_all = None
      self.X_train = None
      self.X_test = None
      self.y_train = None
      self.y_test = None
      self.rseed = 12345

      self.prepare_data()

   # Reads the input pickle file and prepares the data for training
   def prepare_data(self):
      print('*** PREPARE DATA ***')
      # Read the pickle file and select the columns to be used for training and prediction
      df = pd.read_pickle(self.pickle_file)
      cols = ['fraud', 'q', 'currency', 'exchangerate', 'prefcode', 'fta', 'port', 'm_cif_factor', 'm_vat_rate',
              'm_duty_rate', 'm_exciseadv_rate', 'm_tax_rate', 'fx_usd', 'quart', 'subregion']
      df = df[cols]
      print('*** original data ***')
      print(df.describe(include='all'))
      print(df.info())

      # Select categorical features
      df_cat = df.select_dtypes(include=['category'])

      # Encode categorical columns
      for i in df_cat.columns:
         enc = LabelEncoder()
         encoded = enc.fit_transform(df_cat[i])
         df.drop(columns=[i])
         df[i] = encoded

      print('*** encoded data ***')
      print(df.describe(include='all'))
      print(df.info())

      # Split data to train and test datasets
      X = df.drop(columns=['fraud'])
      y = df['fraud']
      split_array = train_test_split(X, y, test_size=0.2, random_state=self.rseed)

      self.X_train, self.X_test, self.y_train, self.y_test = split_array
      self.df_all = df

if __name__ == "__main__":
   CustomsDataModel('./datasets/boc_lite_2017_final2.pkl', output_folder='./model_output')

