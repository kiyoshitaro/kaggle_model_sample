
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import seaborn as sns

def numerize(df):
    # for col in df.columns:
    #     if df[col].dtype == 'object' and len(df[col].unique()) >= 10:
    #         df.drop(columns=[col], inplace=True)
    # df.fillna(0, inplace=True)
    # df = pd.get_dummies(df)
    # return df

    for col in df.columns:
        if df[col].dtype == 'object':
            th = set(df[col])
            if len(th) > 10:
                print("ll")
                df.drop(columns=[col], inplace=True)
            else:
                df[col] = pd.Series([list(th).index(i) for i in df[col]])
    df.fillna(0, inplace=True)
    # df = pd.get_dummies(df)
    return df

if __name__ == '__main__':
    random_state = 42
    np.random.seed(random_state)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    val_aucs = []
    feature_importance_df = pd.DataFrame()
    lgb_params = {
        "objective" : "binary",
        "metric" : "auc",
        "boosting": 'gbdt',
        "max_depth" : -1,
        "num_leaves" : 13,
        "learning_rate" : 0.01,
        "bagging_freq": 5,
        "bagging_fraction" : 0.4,
        "feature_fraction" : 0.05,
        "min_data_in_leaf": 80,
        "min_sum_heassian_in_leaf": 10,
        "tree_learner": "serial",
        "boost_from_average": "false",
        #"lambda_l1" : 5,
        #"lambda_l2" : 5,
        "bagging_seed" : random_state,
        "verbosity" : 1,
        "seed": random_state
    }
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    oof = train_df[['id', 'label']]
    oof['predict'] = 0
    predictions = test_df[['id']]
    df = pd.concat([train_df, test_df])

    df = numerize(df)
    cols=["label","id"]
    train_df = df[df['id'] < 30000]
    test_X = df[df['id'] >= 30000]
    features = [col for col in train_df.columns if col not in ['label', 'id']]
    labels = train_df['label']



    # sc = StandardScaler()
    # pd.DataFrame(sc.fit_transform(train_df))
    # for col in train_df.columns:
    #     train_df[col] = pd.Series(sc.transform(train_df[col]))
    # for col in test_X.columns:
    #     test_X[col] = pd.Series(sc.transform(test_X[col]))


    for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, labels)):
        X_train, y_train = train_df.iloc[trn_idx][features], labels.iloc[trn_idx]
        X_valid, y_valid = train_df.iloc[val_idx][features], labels.iloc[val_idx]
        
        N = 5
        p_valid,yp = 0,0
        for i in range(N):
            X_t, y_t = X_train, y_train
            # X_t = pd.DataFrame(X_t)
            # X_t = X_t.add_prefix('FIELD_')
        
            trn_data = lgb.Dataset(X_t, label=y_t)
            val_data = lgb.Dataset(X_valid, label=y_valid)
            evals_result = {}
            lgb_clf = lgb.train(lgb_params,
                            trn_data,
                            100000,
                            valid_sets = [trn_data, val_data],
                            early_stopping_rounds=3000,
                            verbose_eval = 1000,
                            evals_result=evals_result)
            p_valid += lgb_clf.predict(X_valid)
            yp += lgb_clf.predict(test_X[features])
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = lgb_clf.feature_importance()
        fold_importance_df["fold"] = fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        oof['predict'][val_idx] = p_valid/N
        val_score = roc_auc_score(y_valid, p_valid)
        val_aucs.append(val_score)
        
        predictions['fold{}'.format(fold+1)] = yp/N


    mean_auc = np.mean(val_aucs)
    std_auc = np.std(val_aucs)
    all_auc = roc_auc_score(oof['label'], oof['predict'])
    print("Mean auc: %.9f, std: %.9f. All auc: %.9f." % (mean_auc, std_auc, all_auc))

    cols = (feature_importance_df[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:1000].index)
    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

    plt.figure(figsize=(14,26))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance",ascending=False))
    plt.title('LightGBM Features (averaged over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')

    predictions['label'] = np.mean(predictions[[col for col in predictions.columns if col not in ['id', 'label']]].values, axis=1)
    predictions.to_csv('lgb_all_predictions.csv', index=None)
    sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})
    sub_df["target"] = predictions['target']
    sub_df.to_csv("lgb_submission.csv", index=False)
    oof.to_csv('lgb_oof.csv', index=False)
