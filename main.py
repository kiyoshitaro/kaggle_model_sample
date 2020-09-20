import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from model import Logistic, RandomForestReg, lightgbm_lib
from utils import save_file,describe_data, assessment, correlation_map,srt_box
from matplotlib import pyplot as plt
import seaborn as sns 

train_df = pd.read_csv('houseprice/train.csv')
y = train_df.SalePrice
train_df.drop(["SalePrice"], axis=1, inplace=True)
test_df = pd.read_csv('houseprice/test.csv')
idx = test_df["Id"]
df = pd.concat([train_df, test_df],ignore_index=True)
df.drop(["Id"], axis=1, inplace=True)

# dropping features with predominant values
predominant_features = []
for feature in df.columns:
    predominant_value_count = df[feature].value_counts().max()
    if predominant_value_count / df.shape[0] > 0.995:
        predominant_features.append(feature)
print(predominant_features)
df.drop(predominant_features, axis=1,inplace=True)


# Remove feature that too much nan
categorical_cols, numerical_cols = describe_data(df)
miss_feature = ["Alley","MiscFeature","PoolQC","Fence"]
df.drop(columns=miss_feature,inplace=True)
categorical_cols = [x for x in categorical_cols if x not in miss_feature]


# Visualize category feature

# bar_chart
all_data = pd.concat([df.iloc[:len(y), :], y], axis=1)
fig = plt.figure(figsize=(18,20))
for index in range(len(categorical_cols)):
    # fig = plt.figure(figsize=(18,20))
    plt.subplot(9,5,index+1)
    sns.countplot(x=all_data[categorical_cols].iloc[:,index], data=all_data[categorical_cols].dropna())
    plt.xticks(rotation=90)
fig.tight_layout(pad=1.0)

    
# BOX_PLOT
srt_box('SalePrice', all_data)

# categorical ordinal features
map1 = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
set1 = ['ExterQual', 'ExterCond', 'BsmtQual','BsmtCond', 'HeatingQC',
        'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
for feature in set1:
    df[feature] = df[feature].replace(map1)

map2 = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}
set2 = ['BsmtExposure']
df['BsmtExposure'] = df['BsmtExposure'].replace(map2) 

map3 = {'GLQ': 4,'ALQ': 3,'BLQ': 2,'Rec': 3,'LwQ': 2,'Unf': 1,'None': 0}
set3 = ['BsmtFinType1', 'BsmtFinType2']
for feature in set3:
    df[feature] = df[feature].replace(map3)

map4 = {'Y': 1, 'N': 0}
set4 = ['CentralAir']
df['CentralAir'] = df['CentralAir'].replace(map4)


map5 = {'Typ': 3, 'Min1': 2.5, 'Min2': 2, 'Mod': 1.5, 'Maj1': 1, 'Maj2': 0.5, 'Sev': 0, 'Sal': 0}
set5 = ['Functional']
df['Functional'] = df['Functional'].replace(map5)

#  Creation of new categorical ordinal features
df["TotalGarageQual"] = df["GarageQual"] * df["GarageCond"]
df["TotalExteriorQual"] = df["ExterQual"] * df["ExterCond"]

# for col in df.columns:
#     if df[col].dtype == 'object' and len(df[col].unique()) >= 10:
#         df.drop(columns=[col], inplace=True)

ordinal = list(set(set1)|set(set2)|set(set3)|set(set3)|set(set4)|set(set5))
ordinal.append("TotalGarageQual")
ordinal.append("TotalExteriorQual")

nominal = [i for i in categorical_cols if i not in ordinal]

# Categorical nominal
nominal_data = df[nominal]
df.drop(columns=nominal, inplace=True)


# df['Exterior'] =  df.apply(lambda x: x['Exterior1st'] if (pd.isnull(x['Exterior2nd'])) else str(x['Exterior1st'])+'-'+str(x['Exterior2nd']), axis=1)
# df.drop(['Exterior1st', 'Exterior2nd'],axis=1,inplace=True)

# # Merge 'Condition1', 'Condition2' to 'Condition'
# df['Condition'] =  df.apply(lambda x: x['Condition1'] if (pd.isnull(x['Condition2'])) else str(x['Condition1'])+'-'+str(x['Condition2']), axis=1)
# df.drop(['Condition1', 'Condition2'],axis=1,inplace=True)
df = pd.get_dummies(df).reset_index(drop=True)




# Numerical Feature

# IMPUTE
not_nominal = [x for x in df.columns if x not in nominal]


# from sklearn.impute import SimpleImputer
# imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
# df = pd.DataFrame(imp_mean.fit_transform(df),columns = df.columns)


from sklearn.impute import KNNImputer
imp_knn = KNNImputer(n_neighbors=3, weights="uniform")
# df = pd.DataFrame(imp_knn.fit_transform(df),columns = df.columns)
not_nominal_imputed = pd.DataFrame(imp_knn.fit_transform(df[not_nominal]),columns = df[not_nominal].columns)
df = pd.concat([not_nominal_imputed, df[nominal]],axis = 1)



df = pd.get_dummies(df).reset_index(drop=True)

# VISUALIZE 


# Visualize relation each feature vs y
from scipy.stats import skew
skew_values = df[numerical_cols].apply(lambda x: skew(x))
high_skew = skew_values[skew_values > 0.5]
skew_indices = high_skew.index
for index in skew_indices:
    assessment(pd.concat([df.iloc[:len(y), :], y], axis=1), 'SalePrice', index, -1) #-1 for all data 



# FEATURE SELECTION

# Outliers
# box plots for numerical attributes
# out_col = ['LotFrontage','LotArea','BsmtFinSF1','TotalBsmtSF','GrLivArea']
train = pd.concat( [df[:][:len(y)],y], axis = 1)
test = df[:][len(y):]

fig = plt.figure(figsize=(14,15))
for index,col in enumerate(numerical_cols):
    plt.subplot(len(numerical_cols)//5+1,5,index+1)
    sns.boxplot(y=col, data=train)
fig.tight_layout(pad=1.5)

train = train.drop(train[train['LotFrontage']>200].index,axis = 0).reset_index(drop=True)
train = train.drop(train[train ['LotArea']>100000].index).reset_index(drop=True)
train = train.drop(train[train['BsmtFinSF1']>3000].index).reset_index(drop=True)
train = train.drop(train[train['TotalBsmtSF']>4000].index).reset_index(drop=True)
train = train.drop(train[train['1stFlrSF']>4000].index).reset_index(drop=True)
# train = train.drop(train.LowQualFinSF[train['LowQualFinSF']>550].index)
y = train["SalePrice"]
train.drop(columns=["SalePrice"],inplace=True)
df = pd.concat([train, test]).reset_index(drop=True)


# from sklearn.neighbors import LocalOutlierFactor
# lcf = LocalOutlierFactor(n_neighbors = 20)
# lcf.fit_predict(df)
# x[lcf.negative_outlier_factor_ > self.threshold, :]





# Visualize correlation in pair 
updated_train_set = pd.concat([df[numerical_cols].iloc[:len(y), :], y], axis=1)
correlation_map(updated_train_set, 'SalePrice', 15)
# consider dropping one of the pair elements if signs of colinearity show up

# In highly-correlated attributes choose to drop the column 
# with the lower correlation against SalePrice 
# from the above pairs with more than 0.8 correlation



# FEATURE ENGINEER 

#create some feature using features are highly co-related with SalePrice
df['GrLivArea_2']=df['GrLivArea']**2
df['GrLivArea_3']=df['GrLivArea']**3
df['GrLivArea_4']=df['GrLivArea']**4

df['TotalBsmtSF_2']=df['TotalBsmtSF']**2
df['TotalBsmtSF_3']=df['TotalBsmtSF']**3
df['TotalBsmtSF_4']=df['TotalBsmtSF']**4

df['GarageCars_2']=df['GarageCars']**2
df['GarageCars_3']=df['GarageCars']**3
df['GarageCars_4']=df['GarageCars']**4

df['1stFlrSF_2']=df['1stFlrSF']**2
df['1stFlrSF_3']=df['1stFlrSF']**3
df['1stFlrSF_4']=df['1stFlrSF']**4

df['GarageArea_2']=df['GarageArea']**2
df['GarageArea_3']=df['GarageArea']**3
df['GarageArea_4']=df['GarageArea']**4




df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
# df.rename(columns={"1stFlrSF": "FstFlSF", "2ndFlrSF": "SecFlSF", "3SsnPorch": "ThreeSPorch"}, inplace=True)
# df.drop(columns=['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1,inplace=True)


df['YearsSinceBuilt'] = df['YrSold'].astype(int) - df['YearBuilt']
df['YearsSinceRemod'] = df['YrSold'].astype(int) - df['YearRemodAdd']

df['TotalWalledArea'] = df['TotalBsmtSF'] + df['GrLivArea']
df['TotalPorchArea'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
df['TotalOccupiedArea'] = df['TotalWalledArea'] + df['TotalPorchArea']

# df.drop(['TotalBsmtSF', 'GrLivArea','OpenPorchSF','ThreeSPorch','EnclosedPorch','ScreenPorch','FstFlSF','SecFlSF'],axis=1,inplace=True)


df['OtherRooms'] = df['TotRmsAbvGrd'] - df['BedroomAbvGr'] - df['KitchenAbvGr']
df['TotalBathrooms'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
# df.drop(['FullBath', 'HalfBath','BsmtFullBath','BsmtHalfBath'],axis=1,inplace=True)

df['LotDepth'] = df['LotArea'] / df['LotFrontage']


# Creating new features by using new quality indicators.
# Creating new features  based on previous observations. There might be some highly correlated features now. You cab drop them if you want to...

df['TotalSF'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] +
                       df['1stFlrSF'] + df['2ndFlrSF'])
df['TotalBathrooms'] = (df['FullBath'] +
                              (0.5 * df['HalfBath']) +
                              df['BsmtFullBath'] +
                              (0.5 * df['BsmtHalfBath']))

df['TotalPorchSF'] = (df['OpenPorchSF'] + df['3SsnPorch'] +
                            df['EnclosedPorch'] +
                            df['ScreenPorch'] + df['WoodDeckSF'])

df['YearBlRm'] = (df['YearBuilt'] + df['YearRemodAdd'])

# Merging quality and conditions.

df['TotalExtQual'] = (df['ExterQual'] + df['ExterCond'])
df['TotalBsmQual'] = (df['BsmtQual'] + df['BsmtCond'] +
                            df['BsmtFinType1'] +
                            df['BsmtFinType2'])
df['TotalGrgQual'] = (df['GarageQual'] + df['GarageCond'])
df['TotalQual'] = df['OverallQual'] + df[
    'TotalExtQual'] + df['TotalBsmQual'] + df[
        'TotalGrgQual'] + df['KitchenQual'] + df['HeatingQC']

df['QualGr'] = df['TotalQual'] * df['GrLivArea']
df['QualBsm'] = df['TotalBsmQual'] * (df['BsmtFinSF1'] +
                                                  df['BsmtFinSF2'])
df['QualPorch'] = df['TotalExtQual'] * df['TotalPorchSF']
df['QualExt'] = df['TotalExtQual'] * df['MasVnrArea']
df['QualGrg'] = df['TotalGrgQual'] * df['GarageArea']
df['QlLivArea'] = (df['GrLivArea'] -
                         df['LowQualFinSF']) * (df['TotalQual'])
# df['QualSFNg'] = df['QualGr'] * df['Neighborhood']
    
pd.concat([df,y],axis = 1).to_csv("houseprice/clean_data.csv",index=False)


train_df = df[:len(y)]
test_df = df[len(y):]
# train_df = df[df['Id'] <= len(y)]
# test_df = df[df['Id'] > len(y)]
train_X, val_X, train_y, val_y = train_test_split(train_df, y, train_size=0.99, test_size=0.01,
                                                                random_state=0)


# fitted the RobustScaler on both Train and Test set 
# exposed ourselves to the problem of Data Leakage
# ==> Fit the scaler just on training data, and then transforming it on both training and test data
from sklearn.preprocessing import StandardScaler,RobustScaler
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_df)
val_X = sc.transform(val_X)

# rs = RobustScaler()
# train_X = rs.fit_transform(train_X)
# test_X = rs.transform(test_df)
# val_X = rs.transform(val_X)



from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
rfc_model = RandomForestRegressor(n_estimators=700,max_depth=4,random_state=0,oob_score = True).fit(train_X, train_y)
rfc_model.score(val_X,val_y)
rfc_model.oob_score_

import sklearn
sklearn.metrics.mean_squared_error(val_y,rfc_model.predict(val_X))**1/2


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(train_X, train_y)
y_pred = clf.predict(test_X)


from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



elasticnet_alphas = [5e-5, 1e-4, 5e-4, 1e-3]
elasticnet_l1ratios = [0.8, 0.85, 0.9, 0.95, 1]
ela = ElasticNetCV(max_iter=1e7, alphas=elasticnet_alphas,
                                        cv=kfolds, l1_ratio=elasticnet_l1ratios)
ela.fit(train_X, train_y)
y_pred = ela.predict(test_X)
save_file("houseprice/submission_Elastic.csv",idx,y_pred,"houseprice/sample_submission.csv")




lasso_alphas = [5e-5, 1e-4, 5e-4, 1e-3]
las = LassoCV(max_iter=1e7, alphas=lasso_alphas,
                              random_state=42, cv=kfolds)
las.fit(train_X, train_y)
y_pred = las.predict(test_X)
save_file("houseprice/submission_LassoCV.csv",idx,y_pred,"houseprice/sample_submission.csv")




ridge_alphas = [13.5, 14, 14.5, 15, 15.5]
rid = RidgeCV(alphas=ridge_alphas, cv=kfolds)
rid.fit(train_X, train_y)
y_pred = rid.predict(test_X)
save_file("houseprice/submission_RidgeCV.csv",idx,y_pred,"houseprice/sample_submission.csv")



rng = np.random.RandomState(1)
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)
regr_2.fit(train_X, train_y)
y_pred = regr_2.predict(test_X)
save_file("houseprice/submission_AdaBoost.csv",idx,y_pred,"houseprice/sample_submission.csv")






from sklearn.ensemble import GradientBoostingRegressor
alpha = 0.95
clf = GradientBoostingRegressor(loss='quantile', alpha=alpha,
                                n_estimators=250, max_depth=4,
                                learning_rate=.1, min_samples_leaf=9,
                                min_samples_split=9)

clf.fit(train_X, train_y)
# Make the prediction on the meshed x-axis
y_upper = clf.predict(test_X)
clf.set_params(alpha=1.0 - alpha)
clf.fit(train_X, train_y)
# Make the prediction on the meshed x-axis
y_lower = clf.predict(test_X)
clf.set_params(loss='ls')
clf.fit(train_X, train_y)
# Make the prediction on the meshed x-axis
y_pred = clf.predict(test_X)
save_file("houseprice/submission_GradientBoosting.csv",idx,y_pred,"houseprice/sample_submission.csv")








from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV

xgb1 = XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(train_X,
         train_y)
y_pred = xgb_grid.predict(test_X)
save_file("houseprice/submission_XGBRegressor.csv",idx,y_pred,"houseprice/sample_submission.csv")








from sklearn.model_selection import RandomizedSearchCV
from xgboost.sklearn import XGBRegressor

xgb = XGBRegressor(booster='gbtree', objective='reg:squarederror')
# xgb = XGBRegressor(booster='gbtree', objective='reg:squarederror', colsample_bynode= 0.8,num_parallel_tree = 100,tree_method = 'gpu_hist')

param_lst = {
    'learning_rate' : [0.01, 0.1, 0.15],
    'n_estimators' : [100, 500],
    'max_depth' : [3, 6, 9],
    'min_child_weight' : [1, 5, 10],
    'reg_alpha' : [0.001, 0.01, 0.1],
    'reg_lambda' : [0.001, 0.01, 0.1],
}

xgb_reg = RandomizedSearchCV(estimator = xgb, param_distributions = param_lst,
                              n_iter = 100, scoring = 'neg_root_mean_squared_error',
                              cv = 5)
       
xgb_random = xgb_reg.fit(train_X,
         train_y)

# XGB with tune hyperparameters
best_param = xgb_random.best_params_
xgb = XGBRegressor(**best_param)
xgb.fit(train_X,
         train_y)

y_pred = xgb.predict(test_X)
save_file("houseprice/submission_XGBRegressor_random.csv",idx,y_pred,"houseprice/sample_submission.csv")





# Faster training speed and higher efficiency (use histogram based algorithm i.e it buckets continuous feature values into discrete bins which fasten the training procedure)
# Lower memory usage (Replaces continuous values to discrete bins which result in lower memory usage)
# Better accuracy
# Support of parallel and GPU learning
# Capable of handling large-scale data (capable of performing equally good with large datasets with a significant reduction in training time as compared to XGBOOST)
from lightgbm import LGBMRegressor
lgbm = LGBMRegressor(boosting_type='gbdt',objective='regression', max_depth=-1,
                    lambda_l1=0.0001, lambda_l2=0, learning_rate=0.1,
                    n_estimators=100, max_bin=200, min_child_samples=20, 
                    bagging_fraction=0.75, bagging_freq=5,
                    bagging_seed=7, feature_fraction=0.8,
                    feature_fraction_seed=7, verbose=-1)
param_lst = {
    'max_depth' : [2, 5, 8],
    'learning_rate' : [0.001, 0.01, 0.1],
    'n_estimators' : [300, 500],
    'lambda_l1' : [0.0001, 0.001, 0.01],
    'lambda_l2' : [0, 0.0001, 0.001, 0.01],
    'feature_fraction' : [0.4, 0.6, 0.8],
    'min_child_samples' : [5, 10, 20]
}

lightgbm = RandomizedSearchCV(estimator = lgbm, param_distributions = param_lst,
                              n_iter = 100, scoring = 'neg_root_mean_squared_error',
                              cv = 5)
       
lightgbm_search = lightgbm.fit(train_X,
         train_y)

# LightBGM with tuned hyperparameters
best_param = lightgbm_search.best_params_
lgbm = LGBMRegressor(**best_param)
lgbm.fit(train_X,
         train_y)
y_pred = lgbm.predict(test_X)
save_file("houseprice/submission_lgbm_random.csv",idx,y_pred,"houseprice/sample_submission.csv")





# Catboost itself can deal with categorical features which usually has to be converted to numerical encodings 
# in order to feed into traditional gradient boost frameworks and machine learning models.
# The 2 critical features in Catboost algorithm is the use of ordered boosting and 
# innovative algorithm for processing categorical features, 
# which fight the prediction shift caused by a special kind of target leakage present 
# in all existing implementations of gradient boosting algorithms


from catboost import CatBoostRegressor
cb = CatBoostRegressor(loss_function='RMSE', logging_level='Silent')
# param_lst = {
#     'n_estimators' : [100, 300, 500],
#     'learning_rate' : [0.0001, 0.001, 0.01, 0.1],
#     'l2_leaf_reg' : [0.001, 0.01, 0.1],
#     'random_strength' : [0.25, 0.5 ,1],
#     'max_depth' : [3, 6, 9],
#     'min_child_samples' : [2, 5, 10, 15, 20],
#     'rsm' : [0.5, 0.7, 0.9],
    
# }
param_lst = {
    'n_estimators' : [500],
    'learning_rate' : [0.001, 0.01, 0.1],
    'l2_leaf_reg' : [0.001, 0.01, 0.1],
    'random_strength' : [0.25, 0.5 ,1],
    'max_depth' : [3, 6, 9],
    'min_child_samples' : [2, 5, 10],
    'rsm' : [0.5, 0.7, 0.9],
    
}



catboost = RandomizedSearchCV(estimator = cb, param_distributions = param_lst,
                              n_iter = 100, scoring = 'neg_root_mean_squared_error',
                              cv = 5)


# CatBoost with tuned hyperparams
catboost_search = catboost.fit(train_X,
         train_y)
best_param = catboost_search.best_params_
cb = CatBoostRegressor(**best_param)

cb.fit(train_X,
         train_y)
y_pred = cb.predict(test_X)
save_file("houseprice/submission_catboost_random.csv",idx,y_pred,"houseprice/sample_submission.csv")







from mlxtend.regressor import StackingCVRegressor
stackcv = StackingCVRegressor(regressors=(ela,las,rid,xgb_grid, clf, regr_2),
                              meta_regressor=xgb_grid,
                              use_features_in_secondary=True)
stackcv.fit(np.array(train_X), np.array(train_y))
y_pred = stackcv.predict(test_X)
save_file("houseprice/submission_stackcv.csv",idx,y_pred,"houseprice/sample_submission.csv")






from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(train_X, train_y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(train_X.shape[1]):
    print("%s. feature %d (%f)" % (df.columns[f], indices[f], importances[indices[f]]))

# Plot the impurity-based feature importances of the forest
import matplotlib.pyplot as plt

plt.figure()
plt.title("Feature importances")
plt.bar(range(train_X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(train_X.shape[1]), indices)
plt.xlim([-1, train_X.shape[1]])
plt.show()






import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(rfc_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = df.columns.tolist(), top=150)
y_pred = rfc_model.predict(test_X)



def blend_models_predict(X=X):
    return ((0.1* elastic_model_full_data.predict(X)) + 
            (0.1 * lasso_model_full_data.predict(X)) + 
            (0.05 * ridge_model_full_data.predict(X)) + 
            (0.1 * svr_model_full_data.predict(X)) + 
            (0.1 * gbr_model_full_data.predict(X)) + 
            (0.15 * xgb_model_full_data.predict(X)) + 
            (0.1 * lgb_model_full_data.predict(X)) + 
            (0.3 * stack_gen_model.predict(np.array(X))))
