import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

def describe_data(train_df):
    print("Size : ",train_df.shape)
    categorical_cols = [cname for cname in train_df.columns if train_df[cname].dtype == "object"]
    numerical_cols = [cname for cname in train_df.columns if train_df[cname].dtype in ['int64', 'float64']]
    
    print("There are {} field".format(len(train_df.columns)))
    print("{} categorical_cols:".format(len(categorical_cols)))
    unique = [train_df[t].unique() for t in categorical_cols]
    print([i for i in zip(categorical_cols,
    # unique,
    [len(t) 
    for t in unique])
    ])
    
    print("{} numerical_cols: ".format(len(numerical_cols)))
    print(numerical_cols)

    missing_val_count_by_column_categorical = (train_df[categorical_cols].isnull().sum())
    print("Number of missing values in categorical column:")
    print(missing_val_count_by_column_categorical[missing_val_count_by_column_categorical > 0])

    missing_val_count_by_column_numeric = (train_df[numerical_cols].isnull().sum())
    print("Number of missing values in numerical columns:")
    print(missing_val_count_by_column_numeric[missing_val_count_by_column_numeric > 0])

    return categorical_cols, numerical_cols

def check_importance(X,y,model):
    import eli5
    from eli5.sklearn import PermutationImportance
    perm = PermutationImportance(model, random_state=1).fit(X, y)
    eli5.show_weights(perm, feature_names = X.columns.tolist(), top=150)


def assessment(f_data, f_y_feature, f_x_feature, f_index=-1):
    import seaborn as sns
    """
    Develops and displays a histogram and a scatter plot for a dependent / independent variable pair from
    a dataframe and, optionally, highlights a specific observation on the plot in a different color (red).
    
    Also optionally, if an independent feature is not informed, the scatterplot is not displayed.
    
    Keyword arguments:
    
    f_data      Tensor containing the dependent / independent variable pair.
                Pandas dataframe
    f_y_feature Dependent variable designation.
                String
    f_x_feature Independent variable designation.
                String
    f_index     If greater or equal to zero, the observation denoted by f_index will be plotted in red.
                Integer
    """
    for f_row in f_data:
        if f_index >= 0:
            f_color = np.where(f_data[f_row].index == f_index,'r','g')
            f_hue = None
        else:
            f_color = 'b'
            f_hue = None
    
    f_fig, f_a = plt.subplots(1, 2, figsize=(16,4))
    
    f_chart1 = sns.distplot(f_data[f_x_feature], ax=f_a[0], kde=False, color='r')
    f_chart1.set_xlabel(f_x_feature,fontsize=10)
    
    if f_index >= 0:
        f_chart2 = plt.scatter(f_data[f_x_feature], f_data[f_y_feature], c=f_color, edgecolors='w')
        f_chart2 = plt.xlabel(f_x_feature, fontsize=10)
        f_chart2 = plt.ylabel(f_y_feature, fontsize=10)
    else:

        f_chart2 = sns.regplot(x=f_x_feature,
                    y=f_y_feature,
                    data=f_data,
                    order=3,
                    ci=None,
                    color='#e74c3c',
                    line_kws={'color': 'black'},
                    scatter_kws={'alpha':0.4})


        # f_chart2 = sns.scatterplot(x=f_x_feature, y=f_y_feature, data=f_data, hue=f_hue, legend=False)
        f_chart2.set_xlabel(f_x_feature,fontsize=10)
        f_chart2.set_ylabel(f_y_feature,fontsize=10)

    plt.show()



def correlation_map(f_data, f_feature, f_number):
    # distribute and scatter plots for target versus numerical attributes
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


def srt_box(y, df):
    
    '''A function for displaying categorical variables.'''
    
    fig, axes = plt.subplots(14, 3, figsize=(25, 80))
    axes = axes.flatten()

    for i, j in zip(df.select_dtypes(include=['object']).columns, axes):

        sortd = df.groupby([i])[y].median().sort_values(ascending=False)
        sns.boxplot(x=i,
                    y=y,
                    data=df,
                    palette='plasma',
                    order=sortd.index,
                    ax=j)
        j.tick_params(labelrotation=45)
        j.yaxis.set_major_locator(MaxNLocator(nbins=18))

        plt.tight_layout()

def save_file(file ,id , y_pred, sample):
    cols = pd.read_csv(sample).columns
    res_df = pd.DataFrame({cols[0]: id, cols[1]: y_pred})
    res_df.to_csv(file, index=False)

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