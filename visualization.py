import math
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import os
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

kFRAUD = 'fraud'

class CustomsDataVisualizer:
   def __init__(self, pickle_file, output_folder):
      if not os.path.exists(pickle_file):
         print(f'File not found: {pickle_file}')
         exit()

      self.pickle_file = pickle_file            # Dataset to read and visualize
      self.output_folder = output_folder        # Location of generated visualization files
      self.df_all = None
      self.df_num = None
      self.df_cat = None

      # Create the output folder
      if not os.path.exists(self.output_folder):
         os.mkdir(self.output_folder)

      self.read_data()
      self.summarize_data()
      self.visualize_data()

   def read_data(self):
      self.df_all = pd.read_pickle(self.pickle_file)
      # Select particular features
      cols = ["fraud","p","q","r","r_pct","currency","exchangerate","prefcode","fta",
              "port","m_fob","m_cif","m_cif_factor","m_vat_rate","m_duty_rate",
              "m_exciseadv_rate","m_tax_rate","fx_usd","ty","quart","subregion"]
      self.df_all = self.df_all[cols]
      # Select numeric and categorical features
      self.df_num = self.df_all.select_dtypes(include=[np.number])
      self.df_cat = self.df_all.select_dtypes(include=['category'])

   # Summarizes the data
   def summarize_data(self):
      print(f'SHAPE: {self.df_all.shape}')
      print('*** DATA TYPES ***')
      print(self.df_all.dtypes)
      print('*** DATA PEEK ***')
      print(f'{self.df_all.head(20)}')
      print('*** DATA DESCRIPTION ***')
      print(f'{self.df_all.describe()}')
      print('*** FRAUD CLASS BREAKDOWN ***')
      print(self.df_cat[kFRAUD].value_counts(normalize=True))
      print('*** PEARSON CORRELATION COEFFICIENT FOR NUMERIC FEATURES ***')
      print(self.df_num.corr(method='pearson'))
      print('*** SKEW FOR NUMERIC FEATURES ***')
      print(self.df_num.skew())
      print('*** KURTOSIS FOR NUMERIC FEATURES ***')
      print(self.df_num.kurtosis())

   # Generates visualization graphs
   def visualize_data(self):
      # Categories plotted against fraud count
      pdf = matplotlib.backends.backend_pdf.PdfPages(f'{self.output_folder}/categories.pdf')
      for i in self.df_cat.columns:
         # Adjust width based on number of categories
         cat_num = self.df_cat[i].cat.categories.size
         p = sns.catplot(data=self.df_cat, y=i, kind='count', hue=kFRAUD)
         p.fig.set_size_inches(10, (cat_num * 0.3) + 5, forward=True)
         pdf.savefig()
      pdf.close()

      plt.rcParams['figure.figsize'] = [10, 7]
      layout = (math.ceil((self.df_num.shape[1] - 1) / 5), 5)

      # Histograms
      self.df_num.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
      plt.savefig(f'{self.output_folder}/histogram.png')

      # Density plots
      self.df_num.plot(kind='density', subplots=True, layout=layout, sharex=False, legend=True, fontsize=1)
      plt.savefig(f'{self.output_folder}/density.png')

      # Correlation matrix
      fig = plt.figure()
      ax = fig.add_subplot(111)
      cax = ax.matshow(self.df_num.corr(), vmin=-1, vmax=1, interpolation='none')
      fig.colorbar(cax)
      plt.savefig(f'{self.output_folder}/correlation.png')

      # Box and Whisker plots
      plt.rcParams["figure.figsize"] = [15, 15]
      self.df_num.plot(kind='box', subplots=True, layout=layout, sharex=False, sharey=False)
      plt.savefig(f'{self.output_folder}/boxnwhisker.png')

      # Scatter plot matrix
      plt.rcParams["figure.figsize"] = [40, 40]
      pd.plotting.scatter_matrix(self.df_num)
      plt.savefig(f'{self.output_folder}/scatter.png')


if __name__ == "__main__":
   viz = CustomsDataVisualizer('./datasets/boc_lite_2017_final.pkl', output_folder='./output')
