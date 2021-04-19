import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate, GridSearchCV, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
# Algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)


class CustomsDataModeler:
   def __init__(self, pickle_file, output_folder, nfolds=10):
      if not os.path.exists(pickle_file):
         print(f'File not found: {pickle_file}')
         exit()

      # Create the output folder
      self.output_folder = output_folder        # Location of generated files
      if not os.path.exists(output_folder):
         os.mkdir(output_folder)

      self.pickle_file = pickle_file  # Dataset to read and visualize
      self.nfolds = nfolds
      self.df_all = None
      self.cat_features = list()  # Categorical features
      self.num_features = list()  # Numeric features
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
      self.df_all = df[cols]
      print('*** original data ***')
      print(self.df_all.describe(include='all'))
      print(self.df_all.info())

      # Split data to train and test datasets. Encode the binary 'fraud' target first
      X = self.df_all.drop(columns=['fraud'])
      y = LabelBinarizer().fit_transform(df['fraud']).flatten()

      # One hot encode categorical features and scale numerical features
      self.cat_features = X.select_dtypes(include=['category']).columns
      self.num_features = X.select_dtypes(exclude=['category']).columns
      cat_cnvrtr = OneHotEncoder(drop='first', handle_unknown='error')
      num_cnvrtr = Pipeline([('num_imputer', SimpleImputer(strategy='constant')), ('num_scaler', StandardScaler())])
      col_xfrmr = ColumnTransformer([('cat_xfrmr', cat_cnvrtr, self.cat_features),
                                     ('num_xfrm', num_cnvrtr, self.num_features)])
      X = col_xfrmr.fit_transform(X)
      split_array = train_test_split(X, y, test_size=0.2, random_state=self.rseed)
      self.X_train, self.X_test, self.y_train, self.y_test = split_array

   # Evaluate models
   def evaluate_models(self, X, y, models, scoring_param='roc_auc', lcimage_prefix='learning_curve',
                       scoreimage_prefix='score', timeimage_prefix='time', outimage_prefix='algo_compare',
                       train_sizes=np.linspace(.1, 1.0, 5), n_jobs=-1, seed=1234):
      print('*** EVALUATE MODELS ***')

      plt.rcParams["figure.figsize"] = [10, 10]

      test_score_results = list()
      fit_time_results = list()
      names = list()

      for i, (name, model) in enumerate(models):
         print(f'*** EVALUATING {name} ***')
         kfold = KFold(n_splits=self.nfolds)

         cv_results = cross_validate(model, X, y, cv=kfold, scoring=scoring_param, return_train_score=True)
         train_score = cv_results['train_score']
         test_score = cv_results['test_score']
         fit_time = cv_results['fit_time']
         score_time = cv_results['score_time']
         test_score_results.append(test_score)
         fit_time_results.append(fit_time)
         names.append(name)
         msg = f'{name}: {test_score.mean():.3f} {test_score.std():.3f}'
         print(msg)

         # Generate score and time curve
         plt.clf()
         plt.title(f'Training and Validation Score Curve ({name})')
         plt.xlabel("Split Index")
         plt.ylabel("Score")
         plt.grid()
         plt.plot(range(train_score.size), train_score.tolist(), 'o-', color="g", label=f'Training score ({name})')
         plt.plot(range(test_score.size), test_score.tolist(), 'o-', color="r", label=f'Validation score ({name})')
         plt.legend(loc="best")
         plt.savefig(f'{scoreimage_prefix}_{name}.png')
         plt.clf()

         plt.clf()
         plt.title(f'Fitting and Scoring Time Curve ({name})')
         plt.xlabel("Split Index")
         plt.ylabel("Seconds")
         plt.grid()
         plt.plot(range(fit_time.size), fit_time.tolist(), 'o-', color="g", label=f'Fitting time ({name})')
         plt.plot(range(score_time.size), score_time.tolist(), 'o-', color="r", label=f'Scoring time ({name})')
         plt.legend(loc="best")
         plt.savefig(f'{self.output_folder}/{timeimage_prefix}_{name}.png')
         plt.clf()

         # Generate learning curve
         lc = learning_curve(model, X, y, cv=kfold, n_jobs=n_jobs, train_sizes=train_sizes, shuffle=True)
         train_sizes, train_scores, test_scores = lc
         train_scores_mean = np.mean(train_scores, axis=1)
         train_scores_std = np.std(train_scores, axis=1)
         test_scores_mean = np.mean(test_scores, axis=1)
         test_scores_std = np.std(test_scores, axis=1)

         plt.clf()
         plt.title(f'Learning Curve ({name})')
         plt.xlabel("Training examples")
         plt.ylabel("Score")
         plt.grid()
         plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                          alpha=0.1, color="r")
         plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                          alpha=0.1, color="g")
         plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label=f'Training score ({name})')
         plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label=f'Cross-validation score ({name})')
         plt.legend(loc="best")
         plt.savefig(f'{self.output_folder}/{lcimage_prefix}_{name}.png')
         plt.clf()

      # Compare models
      fig = plt.figure()
      fig.suptitle('Cross Validation Score Comparison')
      ax = fig.add_subplot(111)
      plt.boxplot(test_score_results)
      ax.set_xticklabels(names)
      plt.savefig(f'{self.output_folder}/{outimage_prefix}.png')
      plt.clf()

      fig = plt.figure()
      fig.suptitle('Cross Validation Fit Time Comparison')
      ax = fig.add_subplot(111)
      plt.boxplot(fit_time_results)
      ax.set_xticklabels(names)
      plt.savefig(f'{self.output_folder}/{timeimage_prefix}.png')
      plt.clf()

   # Spot check the models for checking the default performance
   def spot_check_models(self):
      model_list = [#('KNN', KNeighborsClassifier()),
                    # ('SVM', SVC(gamma='auto')),
                    ('DT', DecisionTreeClassifier()),
                    ('RF', RandomForestClassifier())
                    ]
      self.evaluate_models(self.X_train, self.y_train, model_list, lcimage_prefix='spot_lc',
                           scoreimage_prefix='spot_score', timeimage_prefix='spot_time', outimage_prefix='spot')


if __name__ == "__main__":
   modeler = CustomsDataModeler('./datasets/boc_lite_2017_final2.pkl', output_folder='./model_output', nfolds=5)
   modeler.spot_check_models()

