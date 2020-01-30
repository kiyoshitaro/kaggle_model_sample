import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt

def check_importance(X,y):
    import eli5
    from eli5.sklearn import PermutationImportance
    perm = PermutationImportance(rfc_model, random_state=1).fit(X, y)
    eli5.show_weights(perm, feature_names = X.columns.tolist(), top=150)


def check_missing_data(df):
    flag=df.isna().sum().any()
    if flag==True:
        total = df.isnull().sum()
        percent = (df.isnull().sum())/(df.isnull().count()*100)
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        # written by MJ Bahmani
        for col in df.columns:
            dtype = str(df[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return(np.transpose(output))
    else:
        return(False)    


def save_file(file ,id , y_pred):
    res_df = pd.DataFrame({'id': id, 'label': y_pred})
    res_df.to_csv(file, index=False)


def explain(X):
    row_to_show = 5
    data_for_prediction = X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
    rfc_model.predict_proba(data_for_prediction_array)
    import shap  # package used to calculate Shap values
    explainer = shap.TreeExplainer(rfc_model)

    # Calculate Shap values
    shap_values = explainer.shap_values(data_for_prediction)
    shap.initjs()
    shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)


def hist_visual(train_df,feature = "FIELD_22"):
    # train_df['label'].value_counts().plot.bar()

    f,ax=plt.subplots(1,2,figsize=(20,10))
    train_df[train_df['label']==0][feature].plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
    ax[0].set_title('target= 0')
    x1=list(range(0,85,5))
    ax[0].set_xticks(x1)
    train_df[train_df['label']==1][feature].plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
    ax[1].set_title('target= 1')
    x2=list(range(0,85,5))
    ax[1].set_xticks(x2)
    plt.show()




def build_model_input():
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    df = pd.concat([train_df, test_df])
    df.fillna(0, inplace=True)

    for col in df.columns:
        if df[col].dtype == 'object' and len(df[col].unique()) >= 10:
            df.drop(columns=[col], inplace=True)


        # if col == 'FIELD_22':
        #     df[col] = pd.Series([i//10 for i in df[col]])

        # if col == 'FIELD_51':
        #     df[col] = pd.Series([i//10 for i in df[col]])

        # if col == 'FIELD_52':
        #     df[col] = pd.Series([i//1%10 for i in df[col]])

        # if col == 'FIELD_53':
        #     df[col] = pd.Series([i//1%10 for i in df[col]])

        # if col == 'FIELD_54':
        #     df[col] = pd.Series([i*10//1 for i in df[col]])

        # if col == 'FIELD_55':
        #     df[col] = pd.Series([i*10//1 for i in df[col]])

        # if col == 'FIELD_56':
        #     df[col] = pd.Series([i*10//1 for i in df[col]])

        if df[col].dtype == 'object':
            th = set(df[col])
            if len(th) > 70:
                df.drop(columns=[col], inplace=True)
            else:
                df[col] = pd.Series([list(th).index(i) for i in df[col]])
    
    
    # df = pd.get_dummies(df)

    y = df[df['id'] < 30000]['label']
    X = df[df['id'] < 30000].drop(columns=['label'])
    X_pred = df[df['id'] >= 30000]

    return X, X_pred, y