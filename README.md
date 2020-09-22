
## Step

### Preprocess 
- Load data -> combine train and test

- Drop missing-data feature

- Drop predominant feature

- Classify feature -> categorical & numerical


### Clean and creative
- Categorical Feature: 
	- Feature visualize
	- Feature engineer:
		- Encode by hand ordinal feature
		- Use pd.get_dummies to onehot-encode nominal feature


- Numerical Feature:
	- Impute: Use sklearn.impute:  KNNImputer, SimpleImputer,...
	- Visualize feature vs output
	- Visualize relation correlation between features
	- Feature selection: 
		- Consider high-coorrelate features, remove one of them
		- Remove feature that low-coorrelate with output
		- Outlier: Visualize in boxplot and remove with threshold or sklearn.neighbors: LocalOutlierFactor
	- Feature engineer: 
		- Use feature with high coorelate with output to create new feature: log, exp , ...
		- Transform datetime feature to float

- Regularization:
	- Normalize data with sklearn.preprocessing: StandardScaler,RobustScaler, MinMaxScaler . Fit the scaler just on training data, and then transforming it on both training and test data
	- Skew (linear) models love normally distributed data , we need to transform this variable and make it more normally distributed. Using scipy.special: boxcox1p, inv_boxcox, inv_boxcox1p, ... 

### Modeling & evaluate

- Model:
	- Linear model from  sklearn.linear_model : ElasticNetCV, LassoCV, RidgeCV, ...

	- Use ensemble learing from sklearn.ensemble: AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor , HistGradientBoostingRegressor, ...

	- CatBoostRegressor, XGBRegressor, LGBMRegressor

	- Deealearnig model with Keras Sequential

- Validate and blending model:

	- Use model_check in model.py to evaluate model in val_data 

	- Build a normal linear model and fit in train_data to blend all model 

