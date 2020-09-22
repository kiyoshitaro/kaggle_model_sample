
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from model import Logistic, RandomForestReg, lightgbm_lib
from utils import save_file,describe_data, assessment, correlation_map,srt_box
from model import model_check
from matplotlib import pyplot as plt
import seaborn as sns 
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost.sklearn import XGBRegressor



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




# # Categorical nominal
# nominal_data = df[nominal]
# df.drop(columns=nominal, inplace=True)


# df['Exterior'] =  df.apply(lambda x: x['Exterior1st'] if (pd.isnull(x['Exterior2nd'])) else str(x['Exterior1st'])+'-'+str(x['Exterior2nd']), axis=1)
# df.drop(['Exterior1st', 'Exterior2nd'],axis=1,inplace=True)

# # Merge 'Condition1', 'Condition2' to 'Condition'
# df['Condition'] =  df.apply(lambda x: x['Condition1'] if (pd.isnull(x['Condition2'])) else str(x['Condition1'])+'-'+str(x['Condition2']), axis=1)
# df.drop(['Condition1', 'Condition2'],axis=1,inplace=True)
# df = pd.get_dummies(df).reset_index(drop=True)




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




# VISUALIZE 


# Visualize relation each feature vs y
for index in numerical_cols:
    assessment(pd.concat([df.iloc[:len(y), :], y], axis=1), 'SalePrice', index, -1) #-1 for all data 

    

# FEATURE SELECTION

# Outliers
# box plots for numerical attributes
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



# from sklearn.decomposition import PCA, NMF
# pca = PCA().fit(df[numerical_cols])
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')

# n_components = 20
# pca = PCA(n_components=n_components)
# df_pca = pca.fit_transform(df[numerical_cols])

# weights = np.round(pca.components_, 3) 
# ev = np.round(pca.explained_variance_ratio_, 3)
# print('explained variance ratio',ev)
# pca_wt = pd.DataFrame(weights)#, columns=all_data.columns)

# corrmat = pd.DataFrame(df_pca).corr(method='kendall')
# plt.subplots(figsize=(12,9))
# plt.title("Kendall's Correlation Matrix PCA applied", fontsize=16)
# sns.heatmap(corrmat, vmax=0.9, square=True)

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

import datetime
Yr = df['YrSold'].min()
Mo = df['MoSold'].min()
t = datetime.datetime(int(Yr), int(Mo), 1, 0, 0)

def calculateYrMo (row):   
    return int((datetime.datetime(int(row.YrSold),int(row.MoSold),1) - t).total_seconds())
df['YrMoSold'] = df.apply(lambda row: calculateYrMo(row), axis=1)

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
numerical_cols = [i for i in df.columns if df[i].dtype != "object" and i not in ordinal]



# normalize skew feature
from scipy.stats import skew
skew_values = df[numerical_cols].apply(lambda x: skew(x))
high_skew = skew_values[skew_values > 0.5]
skew_indices = high_skew.index
lam_f = 0.15
for index in skew_indices:
    df[index] = boxcox1p(df[index], lam_f)





df = pd.get_dummies(df).reset_index(drop=True)
pd.concat([df,y],axis = 1).to_csv("houseprice/clean_data.csv",index=False)





# TARGET 
from scipy.stats import norm, skew, kurtosis, boxcox #for some statistics
from scipy.special import boxcox1p, inv_boxcox, inv_boxcox1p

sns.distplot(y , fit=norm)
(mu, sigma) = norm.fit(y)
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# The target variable is right skewed. As (linear) models love 
# normally distributed data , we need to transform this variable 
# and make it more normally distributed

# option 1 - original
#y = np.log1p(y)

# Option 2: use box-cox transform - this performs better than the log(1+x)
# try different alpha values  between 0 and 1
lam_l = 0.35 # optimized value
y = boxcox1p(y, lam_l) 

# Option 3: boxcox letting the algorithm select lmbda based on least-likelihood calculation
#y, lam_l = boxcox1p(x=y, lmbda=None)

# option 4 - use log, compare to log1p => score is same
#y = np.log(y)






train_df = df[:len(y)]
test_df = df[len(y):]
train_X, val_X, train_y, val_y = train_test_split(train_df, y, train_size=0.99, test_size=0.01,
                                                                random_state=0)


# fitted the RobustScaler on both Train and Test set 
# exposed ourselves to the problem of Data Leakage
# ==> Fit the scaler just on training data, and then transforming it on both training and test data
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_df)
val_X = sc.transform(val_X)

# rs = RobustScaler()
# train_X = rs.fit_transform(train_X)
# test_X = rs.transform(test_df)
# val_X = rs.transform(val_X)



estimators = []
from sklearn import svm
svm_model=svm.SVC(kernel='rbf').fit(train_X, train_y)
# y_pred = inv_boxcox1p(svm_model.predict(test_X),lam_l)
# save_file("houseprice/submission_svm.csv",idx,y_pred,"houseprice/sample_submission.csv")
estimators.append(svm_model)


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
rfc_model = RandomForestRegressor(n_estimators=700,max_depth=4,random_state=0,oob_score = True).fit(train_X, train_y)
# y_pred = inv_boxcox1p(rfc_model.predict(test_X),lam_l)
# save_file("houseprice/submission_RandomForestRegressor.csv",idx,y_pred,"houseprice/sample_submission.csv")
estimators.append(rfc_model)


rng = np.random.RandomState(1)
from sklearn.ensemble import AdaBoostRegressor 
from sklearn.tree import DecisionTreeRegressor
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng).fit(train_X, train_y)
# y_pred = inv_boxcox1p(regr_2.predict(test_X),lam_l)
# save_file("houseprice/submission_AdaBoost.csv",idx,y_pred,"houseprice/sample_submission.csv")
estimators.append(regr_2)


from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

elasticnet_alphas = [5e-5, 1e-4, 5e-4, 1e-3]
elasticnet_l1ratios = [0.8, 0.85, 0.9, 0.95, 1]
ela = ElasticNetCV(max_iter=1e7, alphas=elasticnet_alphas,
                                        cv=kfolds, l1_ratio=elasticnet_l1ratios).fit(train_X, train_y)
# y_pred = inv_boxcox1p(ela.predict(test_X),lam_l)
# save_file("houseprice/submission_Elastic.csv",idx,y_pred,"houseprice/sample_submission.csv")
estimators.append(ela)


lasso_alphas = [5e-5, 1e-4, 5e-4, 1e-3]
"""This model may be very sensitive to outliers. So we need to made it more robust on them"""
las = LassoCV(max_iter=1e7, alphas=lasso_alphas,
                              random_state=42, cv=kfolds).fit(train_X, train_y)
# y_pred = inv_boxcox1p(las.predict(test_X),lam_l)
# save_file("houseprice/submission_LassoCV.csv",idx,y_pred,"houseprice/sample_submission.csv")
estimators.append(las)


ridge_alphas = [13.5, 14, 14.5, 15, 15.5]
rid = RidgeCV(alphas=ridge_alphas, cv=kfolds).fit(train_X, train_y)
# y_pred = inv_boxcox1p(rid.predict(test_X),lam_l)
# save_file("houseprice/submission_RidgeCV.csv",idx,y_pred,"houseprice/sample_submission.csv")
estimators.append(rid)


from sklearn.ensemble import GradientBoostingRegressor
alpha = 0.95
gbr = GradientBoostingRegressor(loss='quantile', alpha=alpha,
                                n_estimators=250, max_depth=4,
                                learning_rate=.1, min_samples_leaf=9,
                                min_samples_split=9)

gbr.fit(train_X, train_y)
# Make the prediction on the meshed x-axis
y_upper = gbr.predict(test_X)
gbr.set_params(alpha=1.0 - alpha)
gbr.fit(train_X, train_y)
# Make the prediction on the meshed x-axis
y_lower = gbr.predict(test_X)
gbr.set_params(loss='ls')
gbr.fit(train_X, train_y)
# Make the prediction on the meshed x-axis
# y_pred = inv_boxcox1p(gbr.predict(test_X),lam_l)
# save_file("houseprice/submission_GradientBoosting.csv",idx,y_pred,"houseprice/sample_submission.csv")
estimators.append(gbr)


best_param = {'rsm': 0.5,
 'random_strength': 0.5,
 'n_estimators': 500,
 'min_child_samples': 5,
 'max_depth': 3,
 'learning_rate': 0.1,
 'l2_leaf_reg': 0.01}
cb = CatBoostRegressor(**best_param).fit(train_X,train_y)
# y_pred = inv_boxcox1p(cb.predict(test_X),lam_l)
# save_file("houseprice/submission_catboost.csv",idx,y_pred,"houseprice/sample_submission.csv")
estimators.append(cb)


best_param = {'reg_lambda': 0.001,
 'reg_alpha': 0.001,
 'n_estimators': 500,
 'min_child_weight': 5,
 'max_depth': 3,
 'learning_rate': 0.1}
xgb = XGBRegressor(**best_param).fit(train_X,train_y)
# y_pred = inv_boxcox1p(xgb.predict(test_X),lam_l)
# save_file("houseprice/submission_XGBRegressor.csv",idx,y_pred,"houseprice/sample_submission.csv")
estimators.append(xgb)


from sklearn.ensemble import HistGradientBoostingRegressor 
hgrd= HistGradientBoostingRegressor(
    loss= 'least_squares',
    max_depth= 2,
    min_samples_leaf= 40,
    max_leaf_nodes= 29,
    learning_rate= 0.15,
    max_iter= 225,
    random_state=42).fit(train_X,train_y)
# y_pred = inv_boxcox1p(hgrd.predict(test_X),lam_l)
# save_file("houseprice/submission_HistGradientBoostingRegressor.csv",idx,y_pred,"houseprice/sample_submission.csv")


from mlxtend.regressor import StackingCVRegressor
stackcv = StackingCVRegressor(regressors=(ela,las,rid,xgb, clf, regr_2),
                              meta_regressor=xgb,
                              use_features_in_secondary=True)
stackcv.fit(np.array(train_X), np.array(train_y))


from scipy.optimize import minimize
predictions = []
for est in estimators:
    predictions.append(inv_boxcox1p(est.predict(train_X),lam_l))
def mse_func(weights):
    from sklearn.metrics import mean_squared_error
    #scipy minimize will pass the weights as a numpy array
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction
    #return np.mean((y_test-final_prediction)**2)
    return np.sqrt(mean_squared_error(inv_boxcox1p(train_y,lam_l) , final_prediction))
    
starting_values = [0]*len(predictions)
cons = ({'type':'ineq','fun':lambda w: 1-sum(w)})
#our weights are bound between 0 and 1
bounds = [(0,1)]*len(predictions)
res = minimize(mse_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
print('Best Weights: {weights}'.format(weights=res['x']))

weights = res['x']
predictions = []
for est in estimators:
    predictions.append(inv_boxcox1p(est.predict(test_X),lam_l))
y_pred = sum([weight*prediction for weight, prediction in zip(weights, predictions)])
save_file("houseprice/blend_model.csv",idx,y_pred,"houseprice/sample_submission.csv")




# VALIDATE ALL MODEL
raw_models = model_check(train_X, train_y, estimators, kfolds)
pd.options.display.max_columns = 250
pd.options.display.max_rows = 250
pd.options.display(raw_models.style.background_gradient(cmap='summer_r'))





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




import keras
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.layers import Conv1D, BatchNormalization, MaxPool1D, Flatten, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam, SGD, RMSprop   #for adam optimizer
def baseline_model(dim=223, opt_sel="adam", learning_rate = 0.001, neurons = 1, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, decay = 0.0002, momentum=0.9):
    def bm():
        # create model
        model = Sequential()
        #model.add(Dense(neurons, input_dim=223, kernel_initializer='normal', activation='relu'))
        model.add(Dense(neurons, input_dim=dim, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        #model.add(Dense(1, kernel_initializer='normal')) # added to v86
        # Compile model
        if (opt_sel == "adam"):
            #opt = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad) # added to v86
            opt = Adam(lr=learning_rate)
        elif(opt_sel == "sgd"):
            opt = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=opt)
        return model
    return bm
def step_decay(epoch, lr):
    drop = 0.995 # was .999
    epochs_drop = 175.0 # was 175, sgd likes 200+, adam likes 100
    lrate = lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    print("epoch=" + str(epoch) + " lr=" + str(lr) + " lrate=" + str(lrate))
    return lrate

lrate = LearningRateScheduler(step_decay)
early_stopping = EarlyStopping(monitor='val_loss', patience=50, mode='auto', restore_best_weights = True)
callbacks_list = [lrate, early_stopping] 
num_epochs = 100
keras_optimizer = "adam"

# dnn = KerasRegressor(build_fn=baseline_model(dim=223, opt_sel=keras_optimizer, learning_rate = 0.005, neurons = 8, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False), epochs=num_epochs, batch_size=1, verbose=1)
# dnn = KerasRegressor(build_fn=baseline_model(dim=223, opt_sel=keras_optimizer, learning_rate=0.000005, neurons=32, decay=0.000001, momentum=0.9), epochs=num_epochs, batch_size=8, verbose=1)
dnn = KerasRegressor(build_fn=baseline_model(dim=5, opt_sel="adam", learning_rate = 0.001, neurons = 8, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False), epochs=num_epochs, batch_size=2, verbose=1)
dnn.fit(train_X, train_y, shuffle=True, validation_data=(val_X, val_y), callbacks=callbacks_list) # added to v86
dnn_train_pred = inv_boxcox1p(dnn.predict(train_X), lam_l)
dnn_pred = inv_boxcox1p(dnn.predict(test_X), lam_l)
