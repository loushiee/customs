import numpy as np
import pandas as pd
import os

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

kHSCODE = 'hscode'
kCOUNTRY_ORIGIN = 'countryorigin_iso3'
kCOUNTRY_EXPORT = 'countryexport_iso3'
kENTRY = 'entry'
kPREFCODE = 'prefcode'
kCURRENCY = 'currency'
kPORT = 'port'
CATEGORY = 'category'


class DatasetPreprocessor:
   def __init__(self, force_read=False):
      # Pickle file containing all datasets for all years
      self.boc_lite_all_file = './datasets/boc_lite_all_raw.pkl'

      # Years to read
      self.years = np.array([2, 3, 4, 5, 6, 7], dtype=np.int) + 2010

      # If the pickle file is not present then read each csv file and save to the pickle file
      if force_read or not os.path.exists(self.boc_lite_all_file):
         self.df_all = self.read_all_years()
         self.df_all.to_pickle(self.boc_lite_all_file)
      else:
         self.df_all = pd.read_pickle(self.boc_lite_all_file)

      # print('*** dtypes before cleanup ***')
      # print(self.df_all.dtypes)
      # print('*** summary before cleanup ***')
      # print(self.df_all.describe(include='all'))

      self.cleanup()

      print('*** dtypes after cleanup ***')
      print(self.df_all.dtypes)
      print('*** summary after cleanup ***')
      print(self.df_all.describe(include='all'))

   # Reads all the datasets for all years
   def read_all_years(self):
      print('*** read_all_years ***')

      # Read the datasets for all years
      # Data types for the columns
      dtypes = {kHSCODE: CATEGORY,
                kCOUNTRY_ORIGIN: CATEGORY,
                kCOUNTRY_EXPORT: CATEGORY,
                kENTRY: CATEGORY,
                kPREFCODE: CATEGORY,
                kCURRENCY: CATEGORY,
                kPORT: CATEGORY,
                'uid': str,
                'dutiablevaluephp': np.float,
                'dutypaid': np.float,
                'exciseadvalorem': np.float,
                'vatbase': np.float,
                'vatpaid': np.float
                }
      df_list = list()
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
         # print(df_y[kPREFCODE].cat.categories)
         # print(df_y.describe(include='all'))
         # print(f"*** END YEAR {y} ***")

         df_list.append(df_y)

      # Merge all years' data into 1 dataframe
      df_all_years = pd.concat(df_list, copy=False)
      # print(f"*** START ALL YEARS ***")
      # print(df_all.head(n=20))
      # print(f"*** END ALL YEARS ***")

      return df_all_years

   # Cleans up the data based on recommendations
   def cleanup(self):
      if self.df_all.empty:
         print('EMPTY DATASET!!!')
         return

      print('*** CLEANUP ***')
      # 'prefcode' column: change "" values to "NOCODE" and convert 'prefcode' to categorical variable
      self.df_all[kPREFCODE].fillna('NOCODE', inplace=True)
      self.make_categorical(kPREFCODE)

      # Restrict the sample to consumption imports, as opposed to warehousing and transshipment
      # imports, because Philippine duties and taxes are levied on consumption imports under customs law.
      self.make_categorical(kENTRY)
      self.df_all = self.df_all.loc[self.df_all[kENTRY].isin(['C', ''])]
      print(self.df_all[kENTRY].describe())

      # Check that 'hscode' strings are always 11 characters
      print('*** Cleanup hscode column ***')
      print(self.df_all[kHSCODE].str.len().describe())
      # self.df_all[kHSCODE] = self.df_all[kHSCODE].str.pad(11, side='left', fillchar='0')
      self.make_categorical(kHSCODE)
      # 15 Categories to exclude under Valuation Reforms 2014-2015
      # 7207, 7208, 7209, 7210, 7216, 7225, 7227, 7228, 1601, 1602, 3901, 3902, 3903, 3904, 3907
      regex_str = '^(7207|7208|7209|7210|7216|7225|7227|7228|1601|1602|3901|3902|3903|3904|3907)'
      to_drop = self.df_all.loc[self.df_all[kHSCODE].str.contains(regex_str, regex=True)]
      #to_drop.to_csv('./hscode_todrop.csv', columns=[kHSCODE])
      to_drop_indices = to_drop.index
      print(f'shape before cleanup: {self.df_all.shape}, indices count: {len(to_drop_indices)}')
      self.df_all = self.df_all.drop(index=to_drop_indices)
      print(f'shape after cleanup: {self.df_all.shape}')

      # Force other categorical columns
      self.make_categorical(kCOUNTRY_ORIGIN)
      self.make_categorical(kCOUNTRY_EXPORT)
      self.make_categorical(kCURRENCY)
      self.make_categorical(kPORT)

      # Read files used for JOINING homogeneous reference-priced products
      rauch_sitc2 = pd.read_csv('./Rauch_classification_revised.txt', sep='\t',
                                dtype={'sitc4': int, 'con': str, 'lib': str})
      hs2017_sitc2 = pd.read_csv('./HS2017toSITC2ConversionAndCorrelationTables.txt', sep='\t',
                                 dtype={'From HS 2017': str, 'To SITC Rev. 2': int})
      # Inner join 2 read files
      merged_sitc2 = pd.merge(hs2017_sitc2, rauch_sitc2, how='left', left_on='To SITC Rev. 2', right_on='sitc4')
      merged_sitc2.to_csv('./merged_sitc2.csv')
      print(rauch_sitc2.describe(include='all'))
      print(hs2017_sitc2.describe(include='all'))
      print(merged_sitc2.describe(include='all'))
      self.df_all = self.df_all.loc[self.df_all[kENTRY].isin(['C', ''])]

      # *** Utility functions ***

   # For checking if a value is a float
   def check_float(self, value):
      try:
         float(value)
         return np.NaN
      except ValueError:
         return value

   # Make a column's dtype categorical
   def make_categorical(self, col_key):
      self.df_all[col_key] = self.df_all[col_key].astype(CATEGORY)
      print(f'*** make_categorical: {col_key} ***')
      print(self.df_all[col_key].describe())
      print(self.df_all[col_key].cat.categories)


if __name__ == "__main__":
   pp = DatasetPreprocessor(force_read=False)
