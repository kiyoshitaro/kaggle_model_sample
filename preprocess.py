
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from model import Logistic, RandomForest, lightgbm_lib
from utils import save_file


def count_encode(X, categorical_features, normalize=False):
    # X_ = count_encode(train_X,['FIELD_8', 'FIELD_10'])
    print('Count encoding: {}'.format(categorical_features))
    X_ = pd.DataFrame()
    for cat_feature in categorical_features:
        X_[cat_feature] = X[cat_feature].astype(
            'object').map(X[cat_feature].value_counts())
        if normalize:
            X_[cat_feature] = X_[cat_feature] / np.max(X_[cat_feature])
    X_ = X_.add_suffix('_count_encoded')
    if normalize:
        X_ = X_.astype(np.float32)
        X_ = X_.add_suffix('_normalized')
    else:
        X_ = X_.astype(np.uint32)
    return X_

def labelcount_encode(X, categorical_features, ascending=False):
    print('LabelCount encoding: {}'.format(categorical_features))
    X_ = pd.DataFrame()
    for cat_feature in categorical_features:
        cat_feature_value_counts = X[cat_feature].value_counts()
        value_counts_list = cat_feature_value_counts.index.tolist()
        if ascending:
            # for ascending ordering
            value_counts_range = list(
                reversed(range(len(cat_feature_value_counts))))
        else:
            # for descending ordering
            value_counts_range = list(range(len(cat_feature_value_counts)))
        labelcount_dict = dict(zip(value_counts_list, value_counts_range))
        X_[cat_feature] = X[cat_feature].map(
            labelcount_dict)
    X_ = X_.add_suffix('_labelcount_encoded')
    if ascending:
        X_ = X_.add_suffix('_ascending')
    else:
        X_ = X_.add_suffix('_descending')
    X_ = X_.astype(np.uint32)
    return X_


def target_encode(X, X_valid, categorical_features, test_X=None,
                  target_feature='target'):
    print('Target Encoding: {}'.format(categorical_features))
    X_ = pd.DataFrame()
    X_valid_ = pd.DataFrame()
    if test_X is not None:
        X_test_ = pd.DataFrame()
    for cat_feature in categorical_features:
        group_target_mean = X.groupby([cat_feature])[target_feature].mean()
        X_[cat_feature] = X[cat_feature].map(group_target_mean)
        X_valid_[cat_feature] = X_valid[cat_feature].map(group_target_mean)
    X_ = X_.astype(np.float32)
    X_ = X_.add_suffix('_target_encoded')
    X_valid_ = X_valid_.astype(np.float32)
    X_valid_ = X_valid_.add_suffix('_target_encoded')
    if test_X is not None:
        X_test_[cat_feature] = test_X[cat_feature].map(group_target_mean)
        X_test_ = X_test_.astype(np.float32)
        X_test_ = X_test_.add_suffix('_target_encoded')
        return X_, X_valid_, X_test_
    return X_, X_valid_



cat_features = ['province', 'district', 'maCv',
                'FIELD_7', 'FIELD_8', 'FIELD_9',
                'FIELD_10', 'FIELD_13', 'FIELD_17', 
                'FIELD_24', 'FIELD_35', 'FIELD_39', 
                'FIELD_41', 'FIELD_42', 'FIELD_43', 
                'FIELD_44','FIELD_12']

bool_features = ['FIELD_2', 'FIELD_18', 'FIELD_19', 
                'FIELD_20', 'FIELD_23', 'FIELD_25', 
                'FIELD_26', 'FIELD_27', 'FIELD_28', 
                'FIELD_29', 'FIELD_30', 'FIELD_31', 
                'FIELD_36', 'FIELD_37', 'FIELD_38', 
                'FIELD_47', 'FIELD_48', 'FIELD_49']

ordinal_features = ['FIELD_17','FIELD_35','FIELD_41',
                'FIELD_44','FIELD_24']


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
df = pd.concat([train_df, test_df])
# df.fillna(0, inplace=True)
num_features = [col for col in df.columns if col not in cat_features+bool_features + ["id","label"]]


cat_features_l = [col for col in cat_features if len(df[col].unique()) >= 10]
cat_features_n = bool_features
cat_features_n.extend([col for col in cat_features if len(df[col].unique()) < 10])
cat_l = count_encode(df, cat_features_l,True)
k = df[cat_features_n]
k.fillna(0, inplace=True)
cat_n = pd.get_dummies(k)
ordinal = df[ordinal_features]
for col in ordinal.columns:
    th = set(ordinal[col])
    ordinal[col] = pd.Series([list(th).index(i) for i in ordinal[col]])



numerical = df[num_features]
numerical  = numerical.replace(to_replace = "None", value =-2) 
numerical  = numerical.replace(to_replace = ["02 05 08 11","HT", "GD","08 02","05 08 11 02"], value =-1)
numerical.astype(float)

for col in numerical.columns:
    numerical[col + "_missing"] = numerical[col].isna().astype(int)
numerical["age_mean"] = (numerical["age_source1"] + numerical["age_source2"])/2
for col in numerical:
    print(col,numerical[col].dtype)


from sklearn.preprocessing import MinMaxScaler
for col in numerical.columns:
    numerical[col] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(numerical[col])),columns=[col])



from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="mean")
num_col = numerical.columns 
numerical = pd.DataFrame(imp.fit_transform(numerical),columns = numerical.columns)
cat_n.index = numerical.index
cat_l.index = numerical.index
df_processed = pd.concat([numerical, cat_n,cat_l],axis = 1)

# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values="NaN", strategy="mean", axis = 0)
# imputer = imputer.fit(numerical)
# tmp = imputer.transform(numerical)
