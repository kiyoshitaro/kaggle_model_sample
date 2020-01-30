from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

def Logistic(train_X, train_y,test_X):
    clf = LogisticRegression()
    clf.fit(train_X, train_y)
    return clf.predict_proba(test_X)[:, 1]


def RandomForest(train_X, train_y,test_X):
    rfc_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
    return rfc_model.predict_proba(test_X)[:, 1]

def DecisionTree(train_X, train_y,test_X):
    tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)
    return tree_model.predict_proba(test_X)[:, 1]


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
