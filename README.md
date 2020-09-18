
## Step
- Đọc data -> gộp dữ liệu train và test,

- Phân loại feature -> categorical & numerical

- Feature engineer :

	- categorical: encode cho ordinal và nominal feature , tạo feature mới

		- ordinal: 'ExterQual', 'ExterCond', 'BsmtQual','BsmtCond', 'HeatingQC','KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','CentralAir','Functional'

	-	Sử dụng pd.get_dummies để encode nominal feature


	- numerical:

- Tạo feature mới:

	- YearsSinceBuilt = YrSold - YearBuilt : Thời gian từ khi được xây dựng đến lúc bán

	- YearsSinceRemod = YrSold - YearRemodAdd: Thời gian từ khi được tu sửa đến lúc bán

  
  

- Remove các feature nhiều nhiễu

- Impute data
- Data outlier
- Scale data -> standard
- Visualize distributed
- Đưa train data vào các mô hình : RandomForestRegressor, GradientBoostingRegressor, DecisionTreeRegressor, … và predict test data
