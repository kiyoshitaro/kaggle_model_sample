from sklearn.linear_model import LogisticRegression,LinearRegression

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

def Logistic(train_X, train_y,test_X):
    clf = LogisticRegression()
    clf.fit(train_X, train_y)
    return clf.predict_proba(test_X)[:, 1]

def Linear(train_X, train_y,test_X):
    clf = LinearRegression()
    clf.fit(train_X, train_y)
    return clf.predict(test_X)[:]


def RandomForestCls(train_X, train_y,test_X):
    rfc_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
    return rfc_model.predict_proba(test_X)[:, 1]
def RandomForestReg(train_X, train_y,test_X):
    rfc_model = RandomForestRegressor(n_estimators=700,max_depth=4,random_state=0).fit(train_X, train_y)
    
    return rfc_model.predict(test_X)


def AdaBoostDecTreeReg(train_X, train_y,test_X) : 
    rng = np.random.RandomState(1)
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor
    regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                            n_estimators=300, random_state=rng)
    regr_2.fit(train_X, train_y)
    return regr_2.predict(test_X)

def GradientBoostingReg(train_X, train_y,test_X):
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
    return clf.predict(test_X)

def XGBReg(train_X, train_y,test_X):
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
    return xgb_grid.predict(test_X)

def BaggingCls(train_X, train_y,test_X):
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1
    )
    bag_clf.fit(train_X, train_y)
    return bag_clf.predict(test_X)


def GradientBoostingRegEarlyStop(train_X, train_y,test_X):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.ensemble import GradientBoostingRegressor
    X_train, X_val, y_train, y_val = train_test_split(train_X, train_y)
    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
    gbrt.fit(X_train, y_train)
    errors = [mean_squared_error(y_val, y_pred)
    for y_pred in gbrt.staged_predict(X_val)]
    bst_n_estimators = np.argmin(errors)
    gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators)
    gbrt_best.fit(X_train, y_train)
    return gbrt_best.predict(test_X)

def SVM(train_X, train_y,test_X):
    from sklearn import svm
    svm_model=svm.SVC()
    svm_model.fit(train_X, train_y)
    return svm_model.predict(test_X)

def NaiBay(train_X, train_y,test_X):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import MultinomialNB
    gnb=GaussianNB()
    mnb=MultinomialNB()
    gnb.fit(train_X, train_y)
    mnb.fit(train_X, train_y)
    return gnb.predict(test_X)

def DecisionTree(train_X, train_y,test_X):
    tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)
    return tree_model.predict_proba(test_X)[:, 1]


def Catboost(train_X, train_y,test_X):
    from catboost import CatBoostRegressor
    from sklearn.model_selection import RandomizedSearchCV
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
    return cb.predict(test_X)

def lightgbm_lib(train_X, train_y,test_X):
    import lightgbm as lgb
    d_train = lgb.Dataset(train_X, label=train_y)
    params = {}
    params['learning_rate'] = 0.003
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['metric'] = 'binary_logloss'
    params['sub_feature'] = 0.5
    params['num_leaves'] = 10
    params['min_data'] = 50
    params['max_depth'] = 10
    clf = lgb.train(params, d_train, 100)
    return clf.predict(test_X)

def model_check(X, y, estimators, cv):

    
    ''' A function for testing multiple estimators.'''
    
    model_table = pd.DataFrame()

    row_index = 0
    for est in estimators:
        label = est.__class__.__name__
        MLA_name = label
        model_table.loc[row_index, 'Model Name'] = MLA_name

        cv_results = cross_validate(est,
                                    X,
                                    y,
                                    cv=cv,
                                    scoring='neg_root_mean_squared_error',
                                    return_train_score=True,
                                    n_jobs=-1)

        model_table.loc[row_index, 'Train RMSE'] = -cv_results[
            'train_score'].mean()
        model_table.loc[row_index, 'Test RMSE'] = -cv_results[
            'test_score'].mean()
        model_table.loc[row_index, 'Test Std'] = cv_results['test_score'].std()
        model_table.loc[row_index, 'Time'] = cv_results['fit_time'].mean()

        row_index += 1

    model_table.sort_values(by=['Test RMSE'],
                            ascending=True,
                            inplace=True)

    return model_table