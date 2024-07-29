import json
import pandas as pd
import numpy as np

import math
from scipy import stats
from scipy.special import kl_div
from scipy.spatial import distance
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.metrics import matthews_corrcoef,pairwise_distances

from joblib import Parallel, delayed
from typing import Union, List, Optional,Tuple, Dict, Callable,Any

import pickle
import warnings
warnings.filterwarnings("ignore")

from collections import Counter

import os

from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,precision_score,recall_score,silhouette_score,f1_score,jaccard_score,pairwise
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,mean_squared_log_error,mean_absolute_percentage_error

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder,MinMaxScaler

from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier,KDTree,DistanceMetric,BallTree,KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier,IsolationForest,ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA,KernelPCA
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn import cluster

import lightgbm as lgb


def load_data(data_name,DATA,GMs):

  data_folder=DATA+data_name
  synthetic_dfs={}
  synthetic_dfs['name']=data_name

  with open(f'data/{data_name}/info.json', 'r') as f:
    info = json.load(f)

  column_names = info['column_names']

  c_col=info['cat_columns']
  target_column=info['target_columns']

  if info['task_type']!="regression":
    c_col=c_col+target_column

  n_col= list(set(column_names) - set(c_col))

  synthetic_dfs['c_col']=c_col
  synthetic_dfs['n_col']=n_col
  synthetic_dfs['real']=pd.read_csv(data_folder+'/real.csv')
  synthetic_dfs['test']=pd.read_csv(data_folder+'/test.csv')

  for gm in GMs:
    synthetic_dfs[gm[:-4]]=pd.read_csv(data_folder+'/'+gm)
  return synthetic_dfs

def encode_df(df_raw,c_col):
    df=df_raw.copy()
    cat_dict={}
    for i in c_col:
      df[i]=df[i].astype('category')
      cat_dict[i] = dict(enumerate(df[i].cat.categories))
      df[i]=df[i].cat.codes
      df[i]=df[i].astype('int')

    return df,cat_dict

def data_preprocess(reald,faked,cat_cols,onehot=False):

  real=reald.copy().reset_index(drop=True)
  fake=faked.copy().reset_index(drop=True)

  real,_=encode_df(real,cat_cols)
  fake,_=encode_df(fake,cat_cols)

  ss=MinMaxScaler()
  ohe = OneHotEncoder(sparse=False,handle_unknown="ignore")

  col_names=real.columns.to_list()
  n_col=[s for s in col_names if s not in cat_cols]

  real_scaled=pd.DataFrame(ss.fit_transform(real[n_col]), index=None,columns=n_col)
  fake_scaled=pd.DataFrame(ss.transform(fake[n_col]), index=None,columns=n_col)

  if onehot:

    b_cols=[]
    for i in real.columns:
      if real[i].nunique()==2:
        b_cols.append(i)

    cat_cols=[i for i in cat_cols if i not in b_cols]

    #One-hot-encode the categorical columns.
    ohe_real = ohe.fit_transform(real[cat_cols])
    ohe_fake = ohe.transform(fake[cat_cols])
    #Convert it to df
    real_encoded = pd.DataFrame(ohe_real, index=None,columns=ohe.get_feature_names_out())
    fake_encoded = pd.DataFrame(ohe_fake, index=None,columns=ohe.get_feature_names_out())

    real_processed=pd.concat([real_scaled, real_encoded,real[b_cols]], axis=1)
    fake_processed=pd.concat([fake_scaled, fake_encoded,fake[b_cols]], axis=1)
  else:
    real_processed=pd.concat([real_scaled, real[cat_cols]], axis=1)[col_names]
    fake_processed=pd.concat([fake_scaled, fake[cat_cols]], axis=1)[col_names]

  return real_processed,fake_processed


  # Distance
def get_frequency(X_gt: pd.DataFrame, X_synth: pd.DataFrame, n_histogram_bins: int = 10) -> dict:
    """Get percentual frequencies for each possible real categorical value.

    Returns:
        The observed and expected frequencies (as a percent).
    """
    res = {}
    for col in X_gt.columns:
        local_bins = min(n_histogram_bins, len(X_gt[col].unique()))

        if len(X_gt[col].unique()) < n_histogram_bins:  # categorical
            gt = (X_gt[col].value_counts() / len(X_gt)).to_dict()
            synth = (X_synth[col].value_counts() / len(X_synth)).to_dict()
        else:
            gt_vals, bins = np.histogram(X_gt[col], bins=local_bins)
            synth_vals, _ = np.histogram(X_synth[col], bins=bins)
            gt = {k: v / (sum(gt_vals) + 1e-8) for k, v in zip(bins, gt_vals)}
            synth = {k: v / (sum(synth_vals) + 1e-8) for k, v in zip(bins, synth_vals)}

        for val in gt:
            if val not in synth or synth[val] == 0:
                synth[val] = 1e-11
        for val in synth:
            if val not in gt or gt[val] == 0:
                gt[val] = 1e-11

        if gt.keys() != synth.keys():
            raise ValueError(f"Invalid features. {gt.keys()}. syn = {synth.keys()}")
        res[col] = (list(gt.values()), list(synth.values()))

    return res

def kl_divergence(colname,x,y):
    relative_entropy=np.sum(kl_div(x,y))
    return {'col_name': colname, 'kl_divergence':relative_entropy}

def js_divergence(colname,x,y):
    js=distance.jensenshannon(x,y)
    return {'col_name': colname, 'js_distance': js}

def num_divergence_df(real: pd.DataFrame, fake: pd.DataFrame,stats_func: Callable, numerical_columns=None):

    freqs=get_frequency(real,fake)
    res = {}
    for col in real.columns:
        real_freq, fake_freq = freqs[col]
        res[col]=stats_func(col,real_freq,fake_freq)
    distances_df=pd.DataFrame(res).T.set_index('col_name')
    # distances_df.loc['mean']=distances_df.mean()
    return distances_df.mean()

def em_distance(colname,x,y):
    em=stats.wasserstein_distance(x,y)
    return {'col_name': colname, 'em_distance': em}

def num_distance_df(real: pd.DataFrame, fake: pd.DataFrame, stats_func: Callable, numerical_columns=None) -> List[Dict[str, Any]]:
    assert real.columns.tolist() == fake.columns.tolist(), f'Colums are not identical between `real` and `fake`. '
    if numerical_columns is None:
      numerical_columns=real.columns.tolist()
    real_iter = real[numerical_columns].items()
    fake_iter = fake[numerical_columns].items()
    distances = Parallel(n_jobs=-1)(delayed(stats_func) (colname, real_col.values, fake_col.values) for (colname, real_col), (_, fake_col) in zip(real_iter, fake_iter))
    distances_df = pd.DataFrame(distances).set_index('col_name')
    # distances_df.loc['mean']=distances_df.mean()
    return distances_df.mean()

def uni_test(real,fake,c_col,n_col):
  kl=num_divergence_df(real,fake,kl_divergence)
  js=num_divergence_df(real,fake,js_divergence)
  em=num_distance_df(real,fake,em_distance)
  return [kl,js,em]



# Bivariate
#####Nom to Nom (Cat to Cat) stats############
def conditional_entropy(x,y):
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

def theils_u(x,y):
    """
    also referred to as the Uncertainty Coefficient, is based on the conditional entropy between x and y —
    given the value of x, how many possible states does y have, and how often do they occur. Just like Cramer’s V, the output value is on the range of [0,1]
    asymmetric, meaning U(x,y)≠U(y,x) (while V(x,y)=V(y,x), where V is Cramer’s V)
    """
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = stats.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

def cramers_v(x, y):
    """
    Calculates Cramer's V statistic for categorical-categorical association.
    This is a symmetric coefficient: V(x,y) = V(y,x)
    Parameters:
    -----------
    x : list / NumPy ndarray / Pandas Series  A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series  A sequence of categorical measurements
    bias_correction : Boolean, default = True   Use bias correction from Bergsma and Wicher,Journal of the Korean Statistical Society 42 (2013): 323-328.
    Returns:
    --------
    float in the range of [0,1]
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    v = np.sqrt(phi2 / min(k - 1, r - 1))

    if -0.0001 <= v < 0.0001 or  1. - 0.0001 < v <= 1. + 0.0001:
        rounded_v = 0. if v < 0 else 1.
        # warnings.warn(f'Rounded V = {v} to {rounded_v}. This is probably due to floating point precision issues.',   RuntimeWarning)
        return rounded_v
    else:
        return v


#####Nom to Num stats############
def correlation_ratio(categories,measurements):
    """
    Calculates the Correlation Ratio (sometimes marked by the greek letter Eta)
    for categorical-continuous association. Answers the question - given a continuous value of a measurement, is it
    possible to know which category is it associated with?Value is in the range [0,1], where 0 means a category cannot be determined
    by a continuous measurement, and 1 means a category can be determined with absolute certainty.
    Wikipedia: https://en.wikipedia.org/wiki/Correlation_ratio
    Parameters:
    -----------
    categories : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    measurements : list / NumPy ndarray / Pandas Series
        A sequence of continuous measurements
    Returns:
    --------
    float in the range of [0,1]
    """
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg),
                                      2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        return 0.
    else:
        eta = np.sqrt(numerator / denominator)
        if 1. < eta <= 1.+0.0000:
            warnings.warn(f'Rounded eta = {eta} to 1. This is probably due to floating point precision issues.',
                          RuntimeWarning)
            return 1.
        else:
            return eta

def column_associations(real: pd.DataFrame, c_col : List, theil_u=False) -> pd.DataFrame:

  corr=pd.DataFrame(index=real.columns,columns=real.columns)

  b_col=[]
  m_col=[]
  for i in c_col:
    unique_values = pd.unique(real[i])
    if len(unique_values) == 2:
      b_col.append(i)
    else:
      m_col.append(i)

  for i,ac in enumerate(corr):
    for j, bc in enumerate(corr):
      if i > j:
        continue

      if ac in c_col and bc in c_col:
        if ac in b_col and bc in b_col:
          c=matthews_corrcoef(real[ac].values.astype(int),real[bc].values.astype(int))
        else:
          if theil_u:
            c= theils_u(real[ac].values,real[bc].values)
          else:
            c=cramers_v(real[ac].values,real[bc].values)
      else:
        if ac in b_col or bc in b_col:
          c,_=stats.pointbiserialr(real[ac].values,real[bc].values)
        else:
          if ac in c_col or bc in c_col:
            c=correlation_ratio(real[ac].values,real[bc].values)
          else:
            c, _ = stats.pearsonr(real[ac].sort_values(), real[bc].sort_values())
      corr.loc[ac,bc]=corr.loc[bc,ac]=c
  return corr

def bivariate_test(real,fake,c_col):
  real_corr=column_associations(real,c_col,True)
  fake_corr=column_associations(fake,c_col,True)
  statistics,p=stats.ks_2samp(real_corr.to_numpy().flatten(),fake_corr.to_numpy().flatten())

  return round(p,4),real_corr,fake_corr



##Precision, Recall, Density Coverage

def compute_pairwise_distance(data_x: np.ndarray, data_y: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x

    dists = pairwise_distances(data_x, data_y)
    return dists

def get_kth_value(unsorted: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values

def compute_nearest_neighbour_distances(input_features: np.ndarray, nearest_k: int) -> np.ndarray:
    """
    Args:
        input_features: numpy.ndarray
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii

def compute_prdc(real: np.ndarray, fake: np.ndarray,nearest_k: int) -> Dict:
    """
    Computes precision, recall, density, and coverage given two manifolds.
    Args:
        real: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        dict of precision, recall, density, and coverage.
    """
    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(real, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(fake, nearest_k)
    distance_real_fake = compute_pairwise_distance(real, fake)

    precision = ((distance_real_fake< np.expand_dims(real_nearest_neighbour_distances, axis=1)).any(axis=0).mean())
    recall = ((distance_real_fake< np.expand_dims(fake_nearest_neighbour_distances, axis=0)).any(axis=1).mean())

    density = (1.0 / float(nearest_k)) * (distance_real_fake< np.expand_dims(real_nearest_neighbour_distances, axis=1)).sum(axis=0).mean()

    coverage = (distance_real_fake.min(axis=1) < real_nearest_neighbour_distances).mean()

    return dict(precision=precision, recall=recall, density=density, coverage=coverage)


##Maximum Mean Discrepancy (MMD)
def mmd_kernel(X,Y,kernel='rbf',gamma=1,degree=2,coef0=0):
  """MMD using linear/rbf/polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
  Arguments:
      X {[n_sample1, dim]} -- [X matrix]
      Y {[n_sample2, dim]} -- [Y matrix]
  Keyword Arguments:
      gamma {int} -- [gamma] (default: {1})
      degree {int} -- [degree] (default: {2})
      coef0 {int} -- [constant item] (default: {0})
  Returns:
      [scalar] -- [MMD value]
    """
  if kernel == 'linear':
    delta = X.mean(0) - Y.mean(0)
    score = delta.dot(delta.T)

  elif kernel == 'rbf':
    XX = pairwise.rbf_kernel(X, X, gamma)
    YY = pairwise.rbf_kernel(Y, Y, gamma)
    XY = pairwise.rbf_kernel(X, Y, gamma);
    score=XX.mean() + YY.mean() - 2 * XY.mean()


  elif kernel == 'polynomial':
    XX = pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    score = XX.mean() + YY.mean() - 2 * XY.mean()
  else:
    raise ValueError(f"Unsupported kernel {kernel}")

  return score




## ML Utility
def ml_evaluation(real, fake, c_col,n_col, df_test, target_col: str, target_type: str = 'class'):

    df_test=df_test.copy()

    real=real.copy()
    fake=fake.copy()

    real_x,real_y = real.drop([target_col], axis=1),real[[target_col]]
    fake_x,fake_y = fake.drop([target_col], axis=1),fake[[target_col]]
    X_test,y_test = df_test.drop([target_col], axis=1),df_test[[target_col]]

    if target_type == 'regr':

      estimators = [
          KNeighborsRegressor(),
          lgb.LGBMRegressor(n_estimators=100,random_state=1,verbose=-1),
          Lasso(random_state=1),
          Ridge(alpha=1.0, random_state=1),
          ElasticNet(random_state=1)
      ]

      estimators_names = ['KNN','LGBM','LS','RD','EN']

    elif target_type == 'class':
      estimators = [
          KNeighborsClassifier(),
          lgb.LGBMClassifier(n_estimators=100,random_state=1,verbose=-1),
          RandomForestClassifier(n_estimators=100, random_state=1),
          LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=500, random_state=1),
          DecisionTreeClassifier(random_state=1),
          MLPClassifier([50, 50], solver='adam', activation='relu', learning_rate='adaptive', random_state=1)
      ]

      estimators_names = ['KNN','LGBM','RF','LR','DT','MLP']
      c_col_x=[i for i in c_col if i not in target_col]
    else:
        raise ValueError(f'target_type must be \'regr\' or \'class\'')

    zipped_estimators= zip(estimators_names,estimators)

    R=[]

    # Preprocessing for numerical data
    numerical_transformer = MinMaxScaler()
    # Preprocessing for categorical data
    categorical_transformer =OneHotEncoder(handle_unknown='ignore')
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer,n_col),
            ('cat', categorical_transformer, c_col_x)])

    if target_type == 'class':
        norm=LabelEncoder()
        fake_y=norm.fit_transform(fake_y)
        y_test=norm.transform(y_test)
        for est_name,est in zipped_estimators:
          if(len(np.unique(fake_y)))==1:
            R.append(['data synthesis',est_name,0,0,0,0,0,0])
          else:
            pipe = make_pipeline(preprocessor, est)
            pipe.fit(fake_x,fake_y)
            y_test_probas = pipe.predict_proba(X_test)[:,1]
            y_test_predicted = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_test_predicted)
            pre = precision_score(y_test, y_test_predicted)
            rec = recall_score(y_test, y_test_predicted)
            f1=f1_score(y_test, y_test_predicted)
            aucpr = average_precision_score(y_test, y_test_probas)
            aucroc = roc_auc_score(y_test, y_test_probas)

            R.append(['data synthesis',est_name,acc,pre,rec,f1,aucpr,aucroc])

        dfResults = pd.DataFrame(data=R, columns=['data_strategy','clf_name','acc','pre','rec','f1','aucpr','aucroc']).sort_values(['data_strategy'])

    if target_type == 'regr':
        norm=MinMaxScaler()
        fake_y_s=norm.fit_transform(fake_y)
        y_test=norm.transform(y_test)
        for est_name,est in zipped_estimators:
          if(len(np.unique(fake_y)))==1:
            R.append(['data synthesis',est_name,0,0,0,0,0,0])
          else:
            model= est
            pipe = make_pipeline(preprocessor, est)
            pipe.fit(fake_x.values,fake_y_s)
            y_test_predicted = pipe.predict(X_test.values)
            r2=r2_score(y_test,y_test_predicted)
            mae=mean_absolute_error(y_test,y_test_predicted)
            mse=mean_squared_error(y_test,y_test_predicted,squared=False)

            R.append(['data synthesis',est_name,r2,mae,mse])

        dfResults = pd.DataFrame(data=R, columns=['data_strategy','reg_name','r2','mae','mse']).sort_values(['data_strategy'])

    return dfResults


def ml_detection(real, fake):

  real=real.copy()
  fake=fake.copy()
  real['flag']=1
  fake['flag']=0

  df=pd.concat([real,fake]).reset_index(drop=True)

  x = df.drop(['flag'], axis=1)
  y = df['flag']

  # For reproducibilty:
  np.random.seed(1)

  estimators = [
          RandomForestClassifier(n_estimators=100, random_state=1),
          DecisionTreeClassifier(random_state=1),
      ]
  estimator_names = [type(clf).__name__ for clf in estimators]

  rows = []
  R_prose={}
  for estimator in estimators:
          assert hasattr(estimator, 'fit')
          assert hasattr(estimator, 'score')

  for i, c in enumerate(estimators):
      c.fit(x,y)

  for classifier, estimator_name in zip(estimators,estimator_names):
    real_proba = classifier.predict_proba(x)[:, 1]
    propensitySE=(real_proba-0.5)**2/0.25
    MSE = np.mean(propensitySE)
    row = {'index': f'{estimator_name}', 'MSE': MSE}
    rows.append(row)

  prospensitymse = pd.DataFrame(rows).set_index('index')

  return prospensitymse.mean()


def record_df(reale,fakee):
  fakec=pd.DataFrame()
  kms=cluster.KMeans(n_clusters=20, random_state=0)
  clt=kms.fit(reale)
  fakec['cluster']=clt.predict(np.array(fakee)).tolist()

  distance_real_fake = compute_pairwise_distance(reale, fakee)
  fakec['distance_mean']=distance_real_fake.mean(axis=1).tolist()
  fakec['distance_1nn']=distance_real_fake.min(axis=1).tolist()

  fakec['distance_to_centroid']=clt.transform(fakee).min(axis=1).tolist()
  # fit the model
  isf = IsolationForest(max_samples='auto',bootstrap=True, random_state=1,n_jobs=-1)
  isf.fit(reale)
  fakec['isf_scores']=[-1*s + 0.5 for s in isf.decision_function(fakee)]

  normalized_fake=fakec.groupby('cluster').transform(lambda x: (x - x.min()) / (x.max()-x.min())).fillna(0)

  return pd.concat([fakec,normalized_fake.rename(columns={"distance_mean": "distance_mean_z",
                                                          "distance_1nn": "distance_1nn_z",
                                                          "distance_to_centroid": "distance_to_centroid_z",
                                                          "isf_scores":"isf_scores_z"})],axis=1)



class SDEvaluator:
    """
    Class for evaluating synthetic data. It is given the real and fake data and allows the user to easily evaluate data with the `evaluate` method.
    Additional evaluations can be done with the different methods of evaluate and the visual evaluation method.
    """
    def __init__(self, dataname: str = 'adult', target_col: str = 'income' , target_type: str = 'class', DATA = None):

        self.dataname = dataname
        self.DATA = DATA
        self.GMs = ['SMOTE.csv','ADASYN.csv','SMOTENC.csv','synthpop.csv','copula.csv','tvae.csv','ctgan.csv','ctabgan.csv','tabddpm.csv','ttvae.csv','ttvae_SMOTE.csv','ttvae_rectangle.csv','ttvae_triangle.csv']
        self.target_col=target_col
        self.target_type=target_type

        self.loaded_data=load_data(self.dataname, self.DATA, self.GMs)
      

    def evaluate(self):
        real=self.loaded_data['real'].sample(1000)
        df_test=self.loaded_data['test']
        c_col=self.loaded_data['c_col']
        n_col=self.loaded_data['n_col']
        target_col=self.target_col
        target_type=self.target_type


        uni_results={}
        biv_results={}
        corrs={}

        mul_results={}
        ml_results={}
        pmse_results={}
        distances={}
        distance_results={}

        result_path=self.DATA+self.dataname+'/result'
        if not os.path.exists(result_path):
          os.makedirs(result_path)
        print('Saving evaluation results to {}'.format(result_path))

        for gm in self.GMs:
          print(gm)
          syn_name = gm[:-4]
          fake = self.loaded_data[syn_name].sample(1000)
          reals,fakes=data_preprocess(real,fake,c_col,onehot=False)

          print('-'*20 + 'Univariate Evaluation' + '-'*20)
          uni_results[syn_name]=uni_test(reals,fakes,c_col,n_col)

          print('-'*20 + 'Bivariate Evaluation' + '-'*20)
          p,rc,fc=bivariate_test(reals,fakes,c_col)
          biv_results[syn_name]=p
          corrs[syn_name]=abs(fc-rc)

          print('-'*20 + 'Multivariate Evaluation' + '-'*20)
          mul_results[syn_name]=compute_prdc(reals,fakes,5)

          print('-'*20 + 'ML Evaluation' + '-'*20)
          ml_results[syn_name]=ml_evaluation(real,fake,c_col,n_col,df_test,target_col,target_type)

          print('-'*20 + 'Privacy Evaluation' + '-'*20)
          pmse_results[syn_name]=ml_detection(reals,fakes)
          distance=record_df(reals,fakes)
          distances[syn_name] = distance
          distance_results[syn_name]=distance.mean()


        pickle.dump(uni_results,open(result_path+'/uni_results.pickle', 'wb'))
        pickle.dump(biv_results,open(result_path+'/biv_results.pickle', 'wb'))
        pickle.dump(corrs,open(result_path+'/corrs.pickle', 'wb'))
        pickle.dump(mul_results,open(result_path+'/mul_results.pickle', 'wb'))
        pickle.dump(ml_results,open(result_path+'/ml_results.pickle', 'wb'))
        pickle.dump(pmse_results,open(result_path+'/pmse_results.pickle', 'wb'))
        pickle.dump(distances,open(result_path+'/distances.pickle', 'wb'))
        pickle.dump(distance_results,open(result_path+'/distance_results.pickle', 'wb'))



def eval_all(args):

  dataname = args.dataname
  DATA=args.syn_path
  with open(f'data/{dataname}/info.json', 'r') as f:
    info = json.load(f)

  column_names = info['column_names'] if info['column_names'] else real_raw.columns.tolist()
  target_col_idx=info['target_col_idx']

  target_col=column_names[target_col_idx[0]]
  target_type='regr' 

  if info['task_type']!="regression":
    target_type='class'


  ev = SDEvaluator(dataname, target_col, target_type,DATA)
  ev.evaluate()
