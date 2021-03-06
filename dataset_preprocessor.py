import numpy as np
import pandas as pd
import os
from pandas.api.types import union_categoricals

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

ENTRY_KEY = 'entry'
PREFCODE_KEY = 'prefcode'


class DatasetPreprocessor:
   def __init__(self, force_read=False):
      # Pickle file containing all datasets for all years
      self.boc_lite_all_file = './datasets/boc_lite_all_raw.pkl'

      # Years to read
      self.years = np.array([2, 3, 4, 5, 6, 7], dtype=np.int) + 2010

      # If the pickle file is not present then read each csv file and save to the pickle file
      if force_read or not os.path.exists(self.boc_lite_all_file):
         self.df_all = self.read_all_years()
      else:
         self.df_all = pd.read_pickle(self.boc_lite_all_file)
      print(self.df_all.dtypes)
      print(self.df_all.describe(include='all'))

   # Utility function for checking if a value is a float
   def check_float(self, value):
      try:
         float(value)
         return np.NaN
      except ValueError:
         return value

   # Reads all the datasets for all years
   def read_all_years(self):
      print('*** read_all_years ***')

      # Read the datasets for all years
      # Data types for the columns
      dtypes = {'hscode': str,
                'countryorigin_iso3': str,
                'countryexport_iso3': str,
                ENTRY_KEY: 'category',
                PREFCODE_KEY: 'category',
                'currency': str,
                'uid': str,
                'port': str,
                'dutiablevaluephp': np.float,
                'dutypaid': np.float,
                'exciseadvalorem': np.float,
                'vatbase': np.float,
                'vatpaid': np.float
                }
      df_list = list()
      #prefcode_cols = list()
      for y in np.nditer(self.years):
         print(f'*** YEAR {y} ***')
         # Read the csv file
         csv = f'./datasets/boc_lite_{y}.csv'
         df_y = pd.read_csv(csv, dtype=dtypes, sep=',', low_memory=False,
                            encoding='latin_1')  # Using utf8 or unicode_escape results in errors
         # df_y.fillna(0, inplace=True)

         # Debug code for finding invalid rows
         # bad = df_y['vatbase'].apply(check_float).dropna()
         # print(bad)
         # for i in bad.index:
         #    bad_row = df_y.iloc[i-1]
         #    print(bad_row)
         #    bad_row = df_y.iloc[i]
         #    print(bad_row)
         #    bad_row = df_y.iloc[i+1]
         #    print(bad_row)

         # Show the first few rows and basic stats
         # print(f"*** START YEAR {y} ***")
         # print(df_y.head(n=20))
         # print(df_y.dtypes)
         # print(df_y[PREFCODE_KEY].cat.categories)
         # print(df_y.describe(include='all'))
         # print(f"*** END YEAR {y} ***")

         df_list.append(df_y)
         #prefcode_cols.append(df_y[PREFCODE_KEY])

      # uc = union_categoricals(prefcode_cols)
      # uc.add_categories('NOCODE', inplace=True)
      # for df_y in df_list:
      #    df_y[PREFCODE_KEY] = pd.Categorical(df_y[PREFCODE_KEY], categories=uc.categories)

      # Merge all years' data into 1 dataframe
      df_all_years = pd.concat(df_list, copy=False)
      # print(f"*** START ALL YEARS ***")
      # print(df_all.head(n=20))
      # print(f"*** END ALL YEARS ***")
      df_all_years.to_pickle(self.boc_lite_all_file)

      # print('*** prefcode categories before ***')
      # print(df_all_years[PREFCODE_KEY].cat.categories)

      return df_all_years

   # Cleans up the data based on recommendations
   def cleanup(self):
      if self.df_all.empty:
         print('EMPTY DATASET!!!')
         return

      print('*** CLEANUP ***')
      # 'prefcode' column: change "" values to "NOCODE" and convert 'prefcode' to categorical variable
      self.df_all[PREFCODE_KEY].fillna('NOCODE', inplace=True)
      self.df_all[PREFCODE_KEY] = self.df_all[PREFCODE_KEY].astype('category')
      print('*** prefcode ***')
      print(self.df_all[PREFCODE_KEY].describe())
      print(self.df_all[PREFCODE_KEY].cat.categories)

      # Restrict the sample to consumption imports, as opposed to warehousing and transshipment
      # imports, because Philippine duties and taxes are levied on consumption imports under customs law.
      self.df_all[ENTRY_KEY] = self.df_all[ENTRY_KEY].astype('category')
      self.df_all = self.df_all.loc[self.df_all[ENTRY_KEY].isin(['C', ''])]
      print('*** entry ***')
      print(self.df_all[ENTRY_KEY].describe())
      print(self.df_all[ENTRY_KEY].cat.categories)

      print(self.df_all.dtypes)


if __name__ == "__main__":
   pp = DatasetPreprocessor(force_read=False)
   pp.cleanup()
