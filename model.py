import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pickle
import random
import tensorflow as tf
import time

# From sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, fbeta_score, make_scorer, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, cross_validate, GridSearchCV, RepeatedStratifiedKFold, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
# Algorithms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
# From Keras/Tensorflow
from tensorflow import keras
from keras import backend as K
# Other ML packages
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.under_sampling import RandomUnderSampler

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)


class CustomsDataModeler:
   def __init__(self, pickle_file, output_folder, nfolds=10, nrepeats=1, oversampling_ratio=None,
                undersampling_ratio=None, rseed=None):
      if not os.path.exists(pickle_file):
         print(f'File not found: {pickle_file}')
         exit()

      # Create the output folder
      self.output_folder = output_folder        # Location of generated files
      if not os.path.exists(output_folder):
         os.mkdir(output_folder)

      self.pickle_file = pickle_file  # Dataset to read and visualize
      self.nfolds = nfolds
      self.nrepeats = nrepeats
      self.oversampling_ratio = oversampling_ratio
      self.undersampling_ratio = undersampling_ratio
      self.df_all = None
      self.cat_features = list()  # Categorical features
      self.num_features = list()  # Numeric features
      self.X_train = None
      self.X_val = None
      self.X_test = None
      self.y_train = None
      self.y_val = None
      self.y_test = None
      self.rseed = rseed if rseed is not None else random.randint(1, 100000)
      print(f'Random seed: {self.rseed}')

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

      # Split data to train, validation and test datasets. 70% train, 10% validation, 20% test
      # Encode the binary 'fraud' target first
      X = self.df_all.drop(columns=['fraud'])
      y = LabelBinarizer().fit_transform(df['fraud']).flatten()
      split_array = train_test_split(X, y, test_size=0.3, stratify=y, random_state=self.rseed)
      self.X_train, self.X_test, self.y_train, self.y_test = split_array
      split_array = train_test_split(self.X_test, self.y_test, test_size=2.0/3.0, stratify=self.y_test, random_state=self.rseed)
      self.X_val, self.X_test, self.y_val, self.y_test = split_array
      print('Train, Validation, Test split')
      print(f'X train: {self.X_train.shape}, y train: {self.y_train.shape}')
      print(f'X validation: {self.X_val.shape}, y validation: {self.y_val.shape}')
      print(f'X test: {self.X_test.shape}, y test: {self.y_test.shape}')

      self.cat_features = X.select_dtypes(include=['category']).columns
      self.num_features = X.select_dtypes(exclude=['category']).columns

      # Perform oversampling to overcome imbalanced nature of the dataset, i.e. ~21% is classified as 'fraud'
      pipe_steps = list()
      if self.oversampling_ratio is not None:
         print(f'Oversampling ratio: {self.oversampling_ratio}')
         cat_features_indices = X.columns.get_indexer(self.cat_features)
         over_sampler = SMOTENC(categorical_features=cat_features_indices, sampling_strategy=self.oversampling_ratio)
         pipe_steps.append(('over_sampler', over_sampler))
      if self.undersampling_ratio is not None:
         print(f'Undersampling ratio: {self.undersampling_ratio}')
         under_sampler = RandomUnderSampler(sampling_strategy=self.undersampling_ratio)
         pipe_steps.append(('under_sampler', under_sampler))
      if len(pipe_steps) > 0:
         sampler_pipe = ImblearnPipeline(pipe_steps)
         self.X_train, self.y_train = sampler_pipe.fit_resample(self.X_train, self.y_train)
         npos_y = np.count_nonzero(self.y_train)
         nneg_y = self.y_train.size - npos_y
         print(f'Target ratio after sampling (positive / negative): {npos_y} / {nneg_y} = {100.0 * npos_y / nneg_y:.3f}')

      # One hot encode categorical features and scale numerical features
      cat_cnvrtr = OneHotEncoder(drop='first', handle_unknown='error')
      num_cnvrtr = Pipeline([('num_imputer', SimpleImputer(strategy='constant')), ('num_scaler', StandardScaler())])
      col_xfrmr = ColumnTransformer([('cat_xfrmr', cat_cnvrtr, self.cat_features),
                                     ('num_xfrm', num_cnvrtr, self.num_features)])
      self.X_train = col_xfrmr.fit_transform(self.X_train).todense()
      self.X_val = col_xfrmr.transform(self.X_val).todense()
      self.X_test = col_xfrmr.transform(self.X_test).todense()

   # Evaluate models
   def evaluate_models(self, X, y, models, metrics=None, lcimage_prefix='learning_curve',
                       scoreimage_prefix='score', timeimage_prefix='time', outimage_prefix='algo_compare',
                       train_sizes=np.linspace(.1, 1.0, 5), n_jobs=-1):
      print('*** EVALUATE MODELS ***')
      if metrics is None:
         metrics = 'f1'

      plt.rcParams["figure.figsize"] = [max(len(models) * 1.5, 10), 10]

      test_score_results = [list() for m in metrics]
      fit_time_results = list()
      names = list()
      metrics_count = len(metrics)

      for i, (name, model) in enumerate(models):
         print(f'*** EVALUATING {name} ***')
         kfold = KFold(n_splits=self.nfolds)
         names.append(name)

         rskf = RepeatedStratifiedKFold(n_splits=self.nfolds, n_repeats=self.nrepeats)
         cv_results = cross_validate(model, X, y, cv=rskf, scoring=metrics, return_train_score=True)
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
            plt.xticks(np.arange(self.nfolds * self.nrepeats))
            plt.ylabel(metrics[i])
            plt.ylim([0, 1])
            plt.grid()
            plt.plot(range(train_scores[i].size), train_scores[i].tolist(), 'o-', color="g", label=f'Training {metrics[i]} ({name})')
            plt.plot(range(test_scores[i].size), test_scores[i].tolist(), 'o-', color="r", label=f'Testing {metrics[i]} ({name})')
            plt.legend(loc="best")
            plt.savefig(f'{self.output_folder}/{scoreimage_prefix}_{name}_{metrics[i]}.png')
            plt.clf()

         plt.clf()
         plt.title(f'Fitting and Scoring Time Curve ({name})')
         plt.xlabel("Split Index")
         plt.xticks(np.arange(self.nfolds * self.nrepeats))
         plt.ylabel("Seconds")
         plt.grid()
         plt.plot(range(fit_time.size), fit_time.tolist(), 'o-', color="g", label=f'Fitting time ({name})')
         plt.plot(range(score_time.size), score_time.tolist(), 'o-', color="r", label=f'Scoring time ({name})')
         plt.legend(loc="best")
         plt.savefig(f'{self.output_folder}/{timeimage_prefix}_{name}.png')
         plt.clf()

         if lcimage_prefix is not None:
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
            plt.fill_between(lc_train_sizes, lc_train_scores_mean - lc_train_scores_std,
                             lc_train_scores_mean + lc_train_scores_std,
                             alpha=0.1, color="r")
            plt.fill_between(lc_train_sizes, lc_test_scores_mean - lc_test_scores_std,
                             lc_test_scores_mean + lc_test_scores_std,
                             alpha=0.1, color="g")
            plt.plot(lc_train_sizes, lc_train_scores_mean, 'o-', color="r", label=f'Training score ({name})')
            plt.plot(lc_train_sizes, lc_test_scores_mean, 'o-', color="g", label=f'Testing score ({name})')
            plt.legend(loc="best")
            plt.savefig(f'{self.output_folder}/{lcimage_prefix}_{name}.png')
            plt.clf()

      # Compare models
      for i in range(metrics_count):
         fig = plt.figure()
         fig.suptitle(f'Testing "{metrics[i]}" Score Comparison')
         ax = fig.add_subplot(111)
         plt.boxplot(test_score_results[i])
         ax.set_xticklabels(names)
         ax.set_ylabel(metrics[i])
         plt.savefig(f'{self.output_folder}/{outimage_prefix}_{metrics[i]}.png')
         plt.clf()

      fig = plt.figure()
      fig.suptitle('Cross Validation Fit Time Comparison')
      ax = fig.add_subplot(111)
      plt.boxplot(fit_time_results)
      ax.set_xticklabels(names)
      ax.set_ylabel('Seconds')
      plt.savefig(f'{self.output_folder}/{timeimage_prefix}.png')
      plt.clf()

   # Perform testing of models using provided test data and target
   def test_models(self, model_list, report_file, use_validation_dataset=False):
      print(f'*** REPORT FILE: {report_file} ***')
      X = self.X_val if use_validation_dataset else self.X_test
      y = self.y_val if use_validation_dataset else self.y_test
      f = open(f'{self.output_folder}/{report_file}', 'w')
      f.write(f'X shape:{X.shape} y shape:{y.shape}\n\n')
      f.write(f'Oversampling ratio: {"None" if self.oversampling_ratio is None else self.oversampling_ratio}\n')
      f.write(f'Undersampling ratio: {"None" if self.undersampling_ratio is None else self.undersampling_ratio}\n\n')

      for i, (name, model) in enumerate(model_list):
         print(f'*** TESTING {name} USING {"VALIDATION" if use_validation_dataset else "TEST"} DATASET ***')
         start = time.time()
         pred = model.predict(X)
         pred = np.where(pred > 0.5, 1, 0)
         runtime = time.time() - start
         f.write(f'*** {name} SUMMARY ***\n')
         f.write(f'{name} Model details:\n{model}\n')
         f.write(f'{name} Runtime (ms): {runtime * 1000.:.3f}\n')
         f.write(f'{name} F1: {f1_score(y, pred) * 100.:.3f}\n')
         f.write(f'{name} F2: {f2_sklearn(y, pred) * 100.:.3f}\n')
         f.write(f'{name} Precision: {precision_score(y, pred) * 100.:.3f}\n')
         f.write(f'{name} Recall: {recall_score(y, pred) * 100.:.3f}\n')
         f.write(f'{name} ROC_AUC: {roc_auc_score(y, pred) * 100.:.3f}\n')
         f.write(f'{name} Balanced Accuracy: {balanced_accuracy_score(y, pred) * 100.:.3f}\n')
         f.write(f'{name} Accuracy: {accuracy_score(y, pred) * 100.:.3f}\n')
         f.write(f'{name} Confusion matrix:\n{confusion_matrix(y, pred)}\n')
         f.write(f'{name} Classification report:\n{classification_report(y, pred)}\n')

   # Tune a model's hyper parameters using grid search
   def grid_search_model(self, model, param_grid, X, y, metrics=None):
      print('*** GRID SEARCH MODEL ***')
      if metrics is None:
         metrics = 'f1'

      rskf = RepeatedStratifiedKFold(n_splits=self.nfolds)
      grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=metrics, cv=rskf,
                          n_jobs=-1, return_train_score=True)
      grid_result = grid.fit(X, y)
      print(f'Non nested best score: {grid_result.best_score_:.3f} using param: {grid_result.best_params_}')
      return grid

   # Spot check the models for checking the default performance
   def spot_check_models(self):
      model_list = [('LinearSVM', LinearSVC(dual=False)),
                    ('LDA', LinearDiscriminantAnalysis()),
                    ('LR_Ridge', LogisticRegression(penalty='l2', solver='liblinear')),
                    ('NaiveBayes', GaussianNB()),
                    ('AdaBoost', AdaBoostClassifier()),
                    ('GradientBoost', GradientBoostingClassifier()),
                    ('XGBoost', XGBClassifier(use_label_encoder=False)),
                    ('NeuralNetwork', MLPClassifier(max_iter=500, early_stopping=True)),
                    ('DecisionTree', DecisionTreeClassifier()),
                    ('RandomForest', RandomForestClassifier()),
                    ]
      self.evaluate_models(self.X_train, self.y_train, model_list,
                           metrics=['roc_auc', 'f1', 'accuracy', 'precision', 'recall'],
                           scoreimage_prefix='spot_score', timeimage_prefix='spot_time', outimage_prefix='spot',
                           lcimage_prefix=None)

   # Tune top performing models from spot checking
   def tune_sklearn_model(self, model, param_grid, metrics, pickle_file=None):
      print('*** MODEL ***')
      print(model)
      print('*** PARAMETERS ***')
      print(param_grid)
      result = self.grid_search_model(model, param_grid, X=self.X_train, y=self.y_train, metrics=metrics)
      print(result.cv_results_)
      tuned_model = result.best_estimator_
      if pickle_file is not None:
         pickle.dump(tuned_model, open(f'{self.output_folder}/{pickle_file}.pkl', 'wb'))
      return tuned_model

   def tune_decision_tree(self, metrics, pickle_file=None):
      print("*** TUNE DECISION TREE ***")
      param_grid = {
         'criterion': ['gini', 'entropy'],
         'max_depth': [15, 16, 17, 18],
         'class_weight': ['balanced', None],
      }
      model = DecisionTreeClassifier()
      return self.tune_sklearn_model(model, param_grid, metrics, pickle_file)

   def tune_random_forest(self, metrics, pickle_file=None):
      print("*** TUNE RANDOM FOREST ***")
      param_grid = {
         'n_estimators': [100, 150, 200],
         'max_depth': [20, 30, 40],
         'max_features': ['auto', 'sqrt'],
      }
      model = RandomForestClassifier(criterion='entropy', class_weight='balanced', oob_score=True, random_state=self.rseed)
      return self.tune_sklearn_model(model, param_grid, metrics, pickle_file)

   def tune_xgboost(self, metrics, scale_pos_weight=1., pickle_file=None):
      print("*** TUNE XGBOOST ***")
      param_grid = {
         'learning_rate': [0.15, 0.2],
         'gamma': [0, 1],
         'max_depth': [15, 20],
         'min_child_weight': [1, 2],
      }
      model = XGBClassifier(scale_pos_weight=scale_pos_weight, tree_method='gpu_hist', objective='binary:logistic',
                            eval_metric='aucpr', use_label_encoder=False, random_state=self.rseed)
      return self.tune_sklearn_model(model, param_grid, metrics, pickle_file)

   def train_neural_network(self, nepochs, class_weight, h5_file):
      print("*** TRAIN NEURAL NETWORK ***")
      nfeatures = self.X_train.shape[1]
      nn_in = keras.Input(shape=(nfeatures,), name='Input')
      regularizer = keras.regularizers.l2(0.02)
      dense1 = keras.layers.Dense(units=nfeatures, activation='relu', kernel_regularizer=regularizer)(nn_in)
      dense2 = keras.layers.Dense(units=nfeatures/2, activation='relu', kernel_regularizer=regularizer)(dense1)
      dense3 = keras.layers.Dense(units=nfeatures/4, activation='relu', kernel_regularizer=regularizer)(dense2)
      nn_out = keras.layers.Dense(units=1, activation='sigmoid', name='Output')(dense3)
      model = keras.Model(inputs=nn_in, outputs=nn_out)
      model.compile(
         optimizer=keras.optimizers.Adam(learning_rate=0.001),
         loss=keras.losses.BinaryCrossentropy(),
         metrics=[keras.metrics.Recall(), keras.metrics.Precision()]
      )
      print(model.summary())
      stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=nepochs/10, verbose=1)
      check_point = keras.callbacks.ModelCheckpoint(f'{self.output_folder}/{h5_file}.h5', monitor='val_recall',
                                                    mode='max', save_best_only=True, verbose=1)
      model.fit(x=self.X_train, y=self.y_train, epochs=nepochs, validation_data=(self.X_val, self.y_val),
                class_weight=class_weight, batch_size=256, callbacks=[stop_early, check_point])
      model = keras.models.load_model(f'{self.output_folder}/{h5_file}.h5')
      return model


# Utility functions for computing F2 score
def f2_sklearn(y_true, y_pred):
   return fbeta_score(y_true, y_pred, beta=2)


def f2_keras(y_true, y_pred):
   def recall(y_t, y_p):
      true_positives = K.sum(K.round(K.clip(y_t * y_p, 0, 1)))
      possible_positives = K.sum(K.round(K.clip(y_t, 0, 1)))
      return (true_positives + K.epsilon()) / (possible_positives + K.epsilon())

   def precision(y_t, y_p):
      true_positives = K.sum(K.round(K.clip(y_t * y_p, 0, 1)))
      predicted_positives = K.sum(K.round(K.clip(y_p, 0, 1)))
      return (true_positives + K.epsilon()) / (predicted_positives + K.epsilon())

   p = precision(y_true, y_pred)
   r = recall(y_true, y_pred)
   return 5 * ((p * r) / ((4 * p) + r + K.epsilon()))


if __name__ == "__main__":
   print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

   # Perform over sampling of minority class (fraud) and undersampling of majority class (non-fraud)
   oversampling_ratio = 35./65.
   undersampling_ratio = 4./6.
   modeler = CustomsDataModeler('./datasets/boc_lite_2017_final2.pkl', output_folder='./model_output',
                                nfolds=5, nrepeats=3, rseed=123456,
                                oversampling_ratio=oversampling_ratio, undersampling_ratio=undersampling_ratio)
   modeler.spot_check_models()

   # Create final models using the best ones from spot checking: Decision Tree, Random Forest, XGBoost and Neural Network
   metric = make_scorer(fbeta_score, beta=2)
   models = list()
   dt = modeler.tune_decision_tree(metrics=metric, pickle_file='decision_tree')
   models.append(('Decision Tree', dt))
   rf = modeler.tune_random_forest(metrics=metric, pickle_file='random_forest')
   models.append(('Random Forest', rf))
   xgb = modeler.tune_xgboost(scale_pos_weight=1./undersampling_ratio, metrics=metric, pickle_file='xgboost')
   models.append(('XGBoost', xgb))
   nn = modeler.train_neural_network(nepochs=1000, class_weight={0: 4, 1: 5}, h5_file='neural_network')
   models.append(('Neural Network', nn))
   modeler.test_models(models, 'model_report_val.txt', use_validation_dataset=True)
   modeler.test_models(models, 'model_report_test.txt', use_validation_dataset=False)
