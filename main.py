import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from model import Logistic, RandomForestReg, lightgbm_lib
from utils import save_file,describe_data, assessment
from matplotlib import pyplot as plt
import seaborn as sns 

train_df = pd.read_csv('houseprice/train.csv')
y = train_df.SalePrice
test_df = pd.read_csv('houseprice/test.csv')
idx = test_df["Id"]
df = pd.concat([train_df, test_df])
df.drop(["SalePrice"], axis=1, inplace=True)


# dropping features with predominant values
predominant_features = []
for feature in df.columns:
    predominant_value_count = df[feature].value_counts().max()
    if predominant_value_count / df.shape[0] > 0.995:
        predominant_features.append(feature)
df.drop(predominant_features, axis=1,inplace=True)



categorical_cols, numerical_cols = describe_data(df)


fig = plt.figure(figsize=(18,16))
for index,col in enumerate(numerical_cols):
    plt.subplot(3,4,index+1)
    sns.distplot(df[col].dropna(), kde=False)
fig.tight_layout(pad=1.0)

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
df.drop(columns=nominal, inplace=True)


df['Exterior'] =  df.apply(lambda x: x['Exterior1st'] if (pd.isnull(x['Exterior2nd'])) else str(x['Exterior1st'])+'-'+str(x['Exterior2nd']), axis=1)
df.drop(['Exterior1st', 'Exterior2nd'],axis=1,inplace=True)

# Merge 'Condition1', 'Condition2' to 'Condition'
df['Condition'] =  df.apply(lambda x: x['Condition1'] if (pd.isnull(x['Condition2'])) else str(x['Condition1'])+'-'+str(x['Condition2']), axis=1)
df.drop(['Condition1', 'Condition2'],axis=1,inplace=True)
df = pd.get_dummies(df).reset_index(drop=True)




# Numerical Feature
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
df = pd.DataFrame(imp_mean.fit_transform(df),columns = df.columns)


from sklearn.impute import KNNImputer
imp_knn = KNNImputer(n_neighbors=3, weights="uniform")
df = pd.DataFrame(imp_knn.fit_transform(df),columns = df.columns)



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



def correlation_map(f_data, f_feature, f_number):
    import seaborn as sns
    from matplotlib import pyplot as plt
    """
    Develops and displays a heatmap plot referenced to a primary feature of a dataframe, highlighting
    the correlation among the 'n' mostly correlated features of the dataframe.
    
    Keyword arguments:
    
    f_data      Tensor containing all relevant features, including the primary.
                Pandas dataframe
    f_feature   The primary feature.
                String
    f_number    The number of features most correlated to the primary feature.
                Integer
    """
    f_most_correlated = f_data.corr().nlargest(f_number,f_feature)[f_feature].index
    f_correlation = f_data[f_most_correlated].corr()
    
    f_mask = np.zeros_like(f_correlation)
    f_mask[np.triu_indices_from(f_mask)] = True
    with sns.axes_style("white"):
        f_fig, f_ax = plt.subplots(figsize=(12, 10))
        f_ax = sns.heatmap(f_correlation, mask=f_mask, vmin=0, vmax=1, square=True,
                           annot=True, annot_kws={"size": 10}, cmap="BuPu")

    plt.show()
    
    
# Visualize correlation in pair 
updated_train_set = pd.concat([df[numerical_cols].iloc[:len(y), :], y], axis=1)
correlation_map(updated_train_set, 'SalePrice', 15)
# consider dropping one of the pair elements if signs of colinearity show up



# Visualize relation each feature vs y
from scipy.stats import skew
skew_values = df.apply(lambda x: skew(x))
high_skew = skew_values[skew_values > 0.5]
skew_indices = high_skew.index
for index in skew_indices:
    assessment(pd.concat([df.iloc[:len(y), :], y], axis=1), 'SalePrice', index, -1)



# Outliers
out_col = ['LotFrontage','LotArea','BsmtFinSF1','TotalBsmtSF','GrLivArea']
fig = plt.figure(figsize=(20,5))
for index,col in enumerate(out_col):
    plt.subplot(1,5,index+1)
    sns.boxplot(y=col, data=X)
fig.tight_layout(pad=1.5)


from sklearn.neighbors import LocalOutlierFactor
lcf = LocalOutlierFactor(n_neighbors = 20)
lcf.fit_predict(df)
x[lcf.negative_outlier_factor_ > self.threshold, :]



train_df = df[df['Id'] <= len(y)]
test_df = df[df['Id'] > len(y)]
train_X, val_X, train_y, val_y = train_test_split(train_df, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_df)
val_X = sc.transform(val_X)



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

