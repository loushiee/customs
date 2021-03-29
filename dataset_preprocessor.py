import numpy as np
import pandas as pd
import os

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

kHSCODE = 'hscode'
kHSCODE6 = 'hscode6'
kCOUNTRY_ORIGIN = 'countryorigin_iso3'
kCOUNTRY_EXPORT = 'countryexport_iso3'
kENTRY = 'entry'
kPREFCODE = 'prefcode'
kCURRENCY = 'currency'
kPORT = 'port'
kSUBPORT = 'subport'
kTY = 'ty'
kTQ = 'tq'
kTM = 'tm'
kFOB = 'm_fob'
kRPCT = 'r_pct'
kFRAUD = 'fraud'
kSUMV = 'sum_value'
kSUMQ = 'sum_quantity'
CATEGORY = 'category'
OTHERS = 'Others'
UNKNOWN = 'Unknown'

class DatasetPreprocessor:
   def __init__(self, force_read=False, force_cleanup=False, compute_rpct=True):
      # Pickle file containing datasets for all years
      self.boc_lite_all_file = './datasets/boc_lite_all_raw.pkl'
      # Pickle file containing cleaned datasets for all years
      self.boc_lite_cleaned_file = './datasets/boc_lite_cleaned.pkl'
      # Pickle file containing cleaned datasets for all years with computed r_pct and r
      self.boc_lite_rpct_file = './datasets/boc_lite_rpct.pkl'
      
      # Years to read
      self.years = np.array([2, 3, 4, 5, 6, 7], dtype=np.int) + 2010

      if not force_cleanup and not os.path.exists(self.boc_lite_cleaned_file):
         force_cleanup = True

      # If the pickle file is not present then read each csv file and save to the pickle file
      if force_read or not os.path.exists(self.boc_lite_all_file):
         self.df_all = self.read_all_years()
         self.df_all.to_pickle(self.boc_lite_all_file)
         force_cleanup = True
      elif force_cleanup:
         self.df_all = pd.read_pickle(self.boc_lite_all_file)

      # Perform data cleanup
      if force_cleanup:
         print('*** dtypes before cleanup ***')
         print(self.df_all.dtypes)
         print('*** summary before cleanup ***')
         print(self.df_all.describe(include='all'))
         print(self.df_all.shape)
         self.cleanup()
         self.df_all.to_pickle(self.boc_lite_cleaned_file)
         compute_rpct = True
      else:
         self.df_all = pd.read_pickle(self.boc_lite_cleaned_file)
      print('*** dtypes after cleanup ***')
      print(self.df_all.dtypes)
      print('*** summary after cleanup ***')
      print(self.df_all.describe(include='all'))
      print(self.df_all.shape)

      # Add the r and r_pct columns
      if compute_rpct or not os.path.exists(self.boc_lite_rpct_file):
         self.add_rpct()
         self.df_all.to_pickle(self.boc_lite_rpct_file)
      else:
         self.df_all = pd.read_pickle(self.boc_lite_rpct_file)
      print('*** summary after rpct ***')
      print(self.df_all.describe(include='all'))
      print(self.df_all.shape)

      self.add_fraud_and_rates()

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
      # Force categorical columns
      self.make_categorical(kCOUNTRY_ORIGIN)
      self.make_categorical(kCOUNTRY_EXPORT)
      self.make_categorical(kCURRENCY)
      self.make_categorical(kTY)
      self.make_categorical(kTQ)
      self.make_categorical(kTM)

      # 'prefcode' column: change "" values to "NOCODE" and convert 'prefcode' to categorical variable
      self.df_all[kPREFCODE].fillna('NOCODE', inplace=True)
      self.make_categorical(kPREFCODE)

      # Restrict the sample to consumption imports, as opposed to warehousing and transshipment
      # imports, because Philippine duties and taxes are levied on consumption imports under customs law.
      self.make_categorical(kENTRY)
      self.df_all = self.df_all.loc[self.df_all[kENTRY].isin(['C', ''])]
      print(self.df_all[kENTRY].describe())

      # hscode column cleanup
      # Check that 'hscode' strings are always 11 characters
      print('*** Cleanup hscode column ***')
      print(self.df_all[kHSCODE].str.len().describe())
      # self.df_all[kHSCODE] = self.df_all[kHSCODE].str.pad(11, side='left', fillchar='0')
      # Create a new hscode_6 column which is the first 6 digits of the hscode
      self.df_all[kHSCODE6] = self.df_all[kHSCODE].str.slice(stop=6)
      self.make_categorical(kHSCODE)
      self.make_categorical(kHSCODE6)
      # 15 Categories to exclude under Valuation Reforms 2014-2015
      # 7207, 7208, 7209, 7210, 7216, 7225, 7227, 7228, 1601, 1602, 3901, 3902, 3903, 3904, 3907
      regex_str = '^(7207|7208|7209|7210|7216|7225|7227|7228|1601|1602|3901|3902|3903|3904|3907)'
      matches = self.df_all[kHSCODE].str.contains(regex_str, regex=True)
      to_drop_indices = matches[matches == True].index
      #print(pd.api.types.is_list_like(to_drop_indices))
      print(f'shape before cleanup: {self.df_all.shape}, match count: {to_drop_indices.size}')
      self.df_all = self.df_all.drop(index=to_drop_indices)
      print(f'shape after cleanup: {self.df_all.shape}')
      # Read files used for JOINING homogeneous reference-priced products
      rauch_sitc2 = pd.read_csv('./Rauch_classification_revised.txt', sep='\t',
                                dtype={'sitc4': int, 'con': str, 'lib': str})
      hs2017_sitc2 = pd.read_csv('./HS2017toSITC2ConversionAndCorrelationTables.txt', sep='\t',
                                 dtype={'From HS 2017': str, 'To SITC Rev. 2': int})
      # Inner join 2 read files
      merged_sitc2 = pd.merge(hs2017_sitc2, rauch_sitc2, how='left', left_on='To SITC Rev. 2', right_on='sitc4')
      #merged_sitc2.to_csv('./merged_sitc2.csv')
      print(rauch_sitc2.describe(include='all'))
      print(hs2017_sitc2.describe(include='all'))
      print(merged_sitc2.describe(include='all'))
      merged_str = merged_sitc2['From HS 2017'].str.cat(sep='|')
      regex_str = f'^({merged_str})'
      matches = self.df_all[kHSCODE].str.contains(regex_str, regex=True)
      print(f'shape before cleanup: {self.df_all.shape}, matches count: {matches[matches == True].index.size}')
      self.df_all = self.df_all.loc[matches]
      print(f'shape after cleanup: {self.df_all.shape}')

      # For subport and port columns, change "" values to "Unknown" to indicate a new category
      regex = r'^\s*$'
      port_array = [kPORT, kSUBPORT]
      for p in port_array:
         print(f'*** Cleanup {p} column')
         matches = self.df_all[p].str.contains(regex, regex=True)
         count = matches[matches == True].index.size
         print(f'{p} with empty values: {count}')
         if count > 0:
            self.df_all[kPORT].replace(regex, 'Unknown', regex=True, inplace=True)
         self.make_categorical(p)
      # Let's keep the top 10 ports and group the remaining ports to just one 'Others' port
      # except for empty port values that will be in 'Unknown'
      print('*** filtering ports ***')
      self.df_all[kPORT] = self.df_all[kPORT].cat.add_categories([OTHERS, UNKNOWN])
      top_ports = self.df_all.groupby([kPORT]).size().sort_values(ascending=False)
      print(top_ports)
      print(top_ports.index)
      top_ports_index = top_ports.index[:10]
      print(top_ports_index)
      print('*** port column before filtering ***')
      print(self.df_all[kPORT].describe())
      print(f'shape: {self.df_all.shape}')
      top_10_ports = self.df_all[kPORT].isin(top_ports_index)
      self.df_all.loc[~top_10_ports & self.df_all[kPORT].notna(), kPORT] = OTHERS
      self.df_all.loc[self.df_all[kPORT].isna(), kPORT] = UNKNOWN
      print('*** port column after filtering ***')
      print(self.df_all[kPORT].describe())
      print(f'shape: {self.df_all.shape}')

      print('*** CLEANUP END ***')

   # Compute r and r_pct and add to the cleaned data frame as columns
   def add_rpct(self):
      print('*** COMPUTE RPCT ***')
      # Compute the r and r_pct for the 6 digit hscode, country and ty
      df_rpct = self.df_all.groupby([kHSCODE6, kCOUNTRY_ORIGIN, kTY]).agg(sum_value=(kFOB, sum), sum_quantity=('q', sum))
      df_rpct[kRPCT] = df_rpct[kSUMV] / df_rpct[kSUMQ]
      df_rpct[kRPCT] = df_rpct[kRPCT].fillna(0)  # Some combinations of hscode/country/ty have 0 quantity
      df_rpct['r'] = df_rpct[kRPCT] * 0.7
      df_rpct.drop(columns=[kSUMV, kSUMQ])
      print(df_rpct.head())
      print(df_rpct.index)
      print(df_rpct.describe(include='all'))
      print(df_rpct.shape)

      # Inner join with the cleaned data to add the r and r_pct columns with the latter
      print('*** Inner join to add r and r_pct columns ***')
      df_rpct = df_rpct.drop(columns=[kSUMV, kSUMQ])
      self.df_all = pd.merge(self.df_all, df_rpct, how='inner', left_on=[kHSCODE6, kCOUNTRY_ORIGIN, kTY],
                             left_index=False, right_on=df_rpct.index, right_index=True)
      print(self.df_all.describe(include='all'))
      print(self.df_all.shape)
      print('*** END RPCT ***')

   # Compute each transaction if fraud or not. Compute t, fta, cif_factor, vat_rate, duty_rate, exciseadv_rate
   def add_fraud_and_rates(self):
      print("*** COMPUTER FRAUD AND RATES  ***")
      self.df_all[kFRAUD] = np.where(self.df_all['p'] < self.df_all['r'], 'Y', 'N')
      self.make_categorical(kFRAUD)
      print(self.df_all[kFRAUD].describe())
      print(self.df_all[kFRAUD].value_counts())
      print(self.df_all[kFRAUD].value_counts(normalize=True))


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
   pp = DatasetPreprocessor(force_read=False, force_cleanup=False, compute_rpct=False)
