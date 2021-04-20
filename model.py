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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

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
      X = col_xfrmr.fit_transform(X).todense()
      split_array = train_test_split(X, y, test_size=0.2, random_state=self.rseed)
      self.X_train, self.X_test, self.y_train, self.y_test = split_array

   # Evaluate models
   def evaluate_models(self, X, y, models, metrics=['roc_auc'], lcimage_prefix='learning_curve',
                       scoreimage_prefix='score', timeimage_prefix='time', outimage_prefix='algo_compare',
                       train_sizes=np.linspace(.1, 1.0, 5), n_jobs=-1, seed=1234):
      print('*** EVALUATE MODELS ***')

      plt.rcParams["figure.figsize"] = [10, 10]

      test_score_results = [list() for m in metrics]
      fit_time_results = list()
      names = list()
      metrics_count = len(metrics)

      for i, (name, model) in enumerate(models):
         print(f'*** EVALUATING {name} ***')
         kfold = KFold(n_splits=self.nfolds)
         names.append(name)

         cv_results = cross_validate(model, X, y, cv=kfold, scoring=metrics, return_train_score=True)
         if metrics_count == 1:
            train_scores = [cv_results['train_score']]
            test_scores = [cv_results['test_score']]
            print(f'{name}: {metrics[0]} training mean and std: {train_scores[0].mean():.3f} {train_scores[0].std():.3f}')
            print(f'{name}: {metrics[0]} testing mean and std: {test_scores[0].mean():.3f} {test_scores[0].std():.3f}')
            test_score_results[0].append(test_scores)
         else:
            train_scores = [cv_results[f'train_{m}'] for m in metrics]
            test_scores = [cv_results[f'test_{m}'] for m in metrics]
            for i in range(metrics_count):
               print(f'{name}: {metrics[i]} training mean and std: {train_scores[i].mean():.3f} {train_scores[i].std():.3f}')
               print(f'{name}: {metrics[i]} testing mean and std: {test_scores[i].mean():.3f} {test_scores[i].std():.3f}')
               test_score_results[i].append(test_scores[i])
         fit_time = cv_results['fit_time']
         score_time = cv_results['score_time']
         fit_time_results.append(fit_time)

         # Generate score and time curve
         for i in range(metrics_count):
            plt.clf()
            plt.title(f'Cross Validation "{metrics[i]}" Curves ({name})')
            plt.xlabel('Split Index')
            plt.xticks(np.arange(self.nfolds))
            plt.ylabel(metrics[i])
            plt.grid()
            plt.plot(range(train_scores[i].size), train_scores[i].tolist(), 'o-', color="g", label=f'Training {metrics[i]} ({name})')
            plt.plot(range(test_scores[i].size), test_scores[i].tolist(), 'o-', color="r", label=f'Testing {metrics[i]} ({name})')
            plt.legend(loc="best")
            plt.savefig(f'{self.output_folder}/{scoreimage_prefix}_{name}_{metrics[i]}.png')
            plt.clf()

         plt.clf()
         plt.title(f'Fitting and Scoring Time Curve ({name})')
         plt.xlabel("Split Index")
         plt.xticks(np.arange(self.nfolds))
         plt.ylabel("Seconds")
         plt.grid()
         plt.plot(range(fit_time.size), fit_time.tolist(), 'o-', color="g", label=f'Fitting time ({name})')
         plt.plot(range(score_time.size), score_time.tolist(), 'o-', color="r", label=f'Scoring time ({name})')
         plt.legend(loc="best")
         plt.savefig(f'{self.output_folder}/{timeimage_prefix}_{name}.png')
         plt.clf()

         # Generate learning curve
         lc = learning_curve(model, X, y, cv=kfold, n_jobs=n_jobs, train_sizes=train_sizes, shuffle=True)
         lc_train_sizes, lc_train_scores, lc_test_scores = lc
         lc_train_scores_mean = np.mean(lc_train_scores, axis=1)
         lc_train_scores_std = np.std(lc_train_scores, axis=1)
         lc_test_scores_mean = np.mean(lc_test_scores, axis=1)
         lc_test_scores_std = np.std(lc_test_scores, axis=1)

         plt.clf()
         plt.title(f'Learning Curve ({name})')
         plt.xlabel("Training examples")
         plt.ylabel("Score")
         plt.grid()
         plt.fill_between(lc_train_sizes, lc_train_scores_mean - lc_train_scores_std, lc_train_scores_mean + lc_train_scores_std,
                          alpha=0.1, color="r")
         plt.fill_between(lc_train_sizes, lc_test_scores_mean - lc_test_scores_std, lc_test_scores_mean + lc_test_scores_std,
                          alpha=0.1, color="g")
         plt.plot(lc_train_sizes, lc_train_scores_mean, 'o-', color="r", label=f'Training score ({name})')
         plt.plot(lc_train_sizes, lc_test_scores_mean, 'o-', color="g", label=f'Testing score ({name})')
         plt.legend(loc="best")
         plt.savefig(f'{self.output_folder}/{lcimage_prefix}_{name}.png')
         plt.clf()

      # Compare models
      for i in range(metrics_count):
         fig = plt.figure()
         fig.suptitle(f'Cross Validation "{metrics[i]}" Score Comparison')
         ax = fig.add_subplot(111)
         plt.boxplot(test_score_results[i])
         ax.set_xticklabels(names)
         plt.savefig(f'{self.output_folder}/{outimage_prefix}_{metrics[i]}.png')
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
      model_list = [('LSVM', LinearSVC(dual=False)),
                    ('LDA', LinearDiscriminantAnalysis()),
                    ('LR_Lasso', LogisticRegression(penalty='l1', solver='liblinear')),
                    ('LR_Ridge', LogisticRegression(penalty='l2', solver='liblinear')),
                    ('NB', GaussianNB()),
                    ('XGB', XGBClassifier(use_label_encoder=False)),
                    ('MLP', MLPClassifier(max_iter=500, early_stopping=True)),
                    ('DT', DecisionTreeClassifier()),
                    ('RF', RandomForestClassifier()),
                    ]
      self.evaluate_models(self.X_train, self.y_train, model_list, metrics=['roc_auc', 'f1', 'accuracy'], lcimage_prefix='spot_lc',
                           scoreimage_prefix='spot_score', timeimage_prefix='spot_time', outimage_prefix='spot')


if __name__ == "__main__":
   modeler = CustomsDataModeler('./datasets/boc_lite_2017_final2.pkl', output_folder='./model_output2', nfolds=5)
   modeler.spot_check_models()

