import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


#Understanding the dataset
df=pd.read_csv('house_price_prediction_dataset.csv')

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

numerical_cols=df.select_dtypes(include=['int64','float64']).columns
categorical_cols=df.select_dtypes(include=['object']).columns

print("Numerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)

# print(df.isnull().sum())

print(df.duplicated().sum())

print(df.columns)

print(df["Price"].describe())


#EDA: Exploratory Data Analysis
# 1. Numerical distributions
# df[numerical_cols].hist(figsize=(12, 10))
# plt.tight_layout()
# plt.show()


# # 2. Price distribution
# plt.figure(figsize=(8,5))
# sns.histplot(df["Price"], bins=30, edgecolor='black',kde="True")
# plt.xlabel("Price")
# plt.ylabel("Frequency")
# plt.show()



# 3. Price boxplot
# plt.figure(figsize=(8, 5))
# sns.boxplot(x=df["Price"])
# plt.title("Price Boxplot")
# plt.show()

# Area boxplot
# plt.figure(figsize=(8,5))
# sns.boxplot(x=df["Area_sqft"])
# plt.title("Area Boxplot")
# plt.show()

# 4. Area vs Price
# plt.figure(figsize=(8, 5))
# plt.scatter(df["Area_sqft"], df["Price"])
# plt.xlabel("Area")
# plt.ylabel("Price")
# plt.title("Area vs Price")
# plt.show()


# # 5. Bedrooms vs Price
# plt.figure(figsize=(8, 5))
# sns.boxplot(data=df, x="Bedrooms", y="Price")
# plt.xlabel("Bedrooms")
# plt.ylabel("Price")
# plt.title("Bedrooms vs Price")
# plt.show()


# # 6. Bathrooms vs Price
# plt.figure(figsize=(8, 5))
# sns.boxplot(data=df, x="Bathrooms", y="Price")
# plt.xlabel("Bathrooms")
# plt.ylabel("Price")
# plt.title("Bathrooms vs Price")
# plt.show()


# # 7. Location vs Price
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=df, x="Location", y="Price")
# plt.xlabel("Location")
# plt.ylabel("Price")
# plt.title("Location vs Price")
# plt.xticks(rotation=45)
# plt.show()


# # 8. Number of houses by Location
# plt.figure(figsize=(10, 5))
# sns.countplot(data=df, x="Location")
# plt.xlabel("Location")
# plt.ylabel("Number of Houses")
# plt.title("Number of Houses by Location")
# plt.xticks(rotation=45)
# plt.show()


# # 9. Correlation matrix
# correlation = df[numerical_cols].corr()

# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation, annot=True, cmap="coolwarm")
# plt.title("Correlation Matrix")
# plt.show()

print(df.isnull().sum())

# numerical data use median to fill missing values

for col in numerical_cols:
    df[col]=df[col].fillna(df[col].median())
    
for col in categorical_cols:
    df[col]=df[col].fillna(df[col].mode()[0])

print(df.isnull().sum())


#remove outliers

# sns.boxplot(x=df['Price'])
# plt.show()

Q1=df['Price'].quantile(0.25)
Q3=df['Price'].quantile(0.75)

print("Q1:", Q1)
print("Q3:", Q3)

IQR=Q3-Q1
print("IQR:", IQR)

lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

outliers=df[(df['Price']<lower_limit) | (df['Price']>upper_limit)]
print("Number of outliers:",outliers.shape[0])

#Area_sqft

sns.boxplot(x=df['Area_sqft'])
plt.show()

AQ1=df['Area_sqft'].quantile(0.25)
AQ3=df['Area_sqft'].quantile(0.75)

print("AQ1:", AQ1)
print("AQ3:", AQ3)

AIQR=AQ3-AQ1
print("AIQR:", AIQR)

Alower_limit=AQ1-1.5*AIQR
Aupper_limit=AQ3+1.5*AIQR
print("Lower Limit:", Alower_limit)
print("Upper Limit:", Aupper_limit)

Aoutliers=df[(df['Area_sqft']<Alower_limit) | (df['Area_sqft']>Aupper_limit)]
print("ANumber of outliers:",Aoutliers.shape[0])

# #bedrooms
# sns.boxplot(x=df["Bedrooms"])
# plt.show()

# #bathrooms
# sns.boxplot(x=df["Bathrooms"])
# plt.show()

# #age_years
# sns.boxplot(x=df["Age_years"])
# plt.show()

# By for loop

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    
    IQR = Q3 - Q1
    
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    outliers = df[
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    ]
    
    print("\nColumn:", col)
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)
    print("Number of outliers:", len(outliers))
    
    #Feature Engineering
    
    df["Price_per_sqft"] = df["Price"] / df["Area_sqft"]
    df["Total_Rooms"]= df["Bedrooms"] + df["Bathrooms"]
    df["Area_per_Bedroom"] = df["Area_sqft"] / df["Bedrooms"]
    df["Age_group"] = pd.cut(
        df["Age_years"],
        bins=[0, 10, 20, 40, 100],
        labels=["New", "Moderate", "Old", "Very Old"]
    )    
    print(df.head())
    print(df.columns)
    
    #seperating features(X) and target variable(y)
    X=df.drop(["Price"],axis=1)
    y=df["Price"]
        
    print(X.head());
    print(y.head());
    
    #encoding categorical variables
    X = pd.get_dummies(X, dtype=int)  
    print(df.head())
    print(df.dtypes)
    
   
    #Train-Test Split
    X_train,X_test,y_train,y_test=train_test_split(
        X,y,
        test_size=0.2,
        random_state=42
    )
    
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    # Feature Scaling
    scaler=StandardScaler()
    
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    
    
print("Scaled training shape:", X_train_scaled.shape)
print("Scaled testing shape:", X_test_scaled.shape)


#Model Building "Linear Regression"
model=LinearRegression()

model.fit(X_train_scaled,y_train)

y_pred=model.predict(X_test_scaled)

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

#evaluating the model

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

# Try polynomial degrees 1 to 4
for degree in range(1, 5):
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_rmse = mean_squared_error(y_train, train_pred) ** 0.5
    test_rmse = mean_squared_error(y_test, test_pred) ** 0.5
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    print(f"\nDegree: {degree}")
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Test R2: {test_r2:.4f}")
    
    
# 1. Create the Ridge model
model_ridge = Ridge(alpha=1.0)

# 2. Train the model
model_ridge.fit(X_train, y_train)

# 3. Predict house prices
y_pred_ridge = model_ridge.predict(X_test)

# 4. Evaluate the model
mae = mean_absolute_error(y_test, y_pred_ridge)
mse = mean_squared_error(y_test, y_pred_ridge)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred_ridge)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# 1. Create the Lasso model
model_lasso = Lasso(alpha=1.0, max_iter=10000)

# 2. Train the model
model_lasso.fit(X_train, y_train)

# 3. Predict house prices
y_pred_lasso = model_lasso.predict(X_test)

# 4. Evaluate the model
mae = mean_absolute_error(y_test, y_pred_lasso)
mse = mean_squared_error(y_test, y_pred_lasso)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred_lasso)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)


# 1. Create the ElasticNet model
model_elastic = ElasticNet(
    alpha=1.0,
    l1_ratio=0.5,
    max_iter=10000
)

# 2. Train the model
model_elastic.fit(X_train, y_train)

# 3. Predict house prices
y_pred_elastic = model_elastic.predict(X_test)


# 1. Enter details of a new house
new_house = pd.DataFrame([{
    "Area_sqft": 1200,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Location": "Pune",
    "Age_years": 5,
    "Parking": 1,
    "Floor": 4,
    "Condition": "Good"
}])

# 2. Apply the same feature engineering
new_house["Total_Rooms"] = (
    new_house["Bedrooms"] + new_house["Bathrooms"]
)

new_house["Area_per_Bedroom"] = (
    new_house["Area_sqft"] / new_house["Bedrooms"]
)

# 3. Encode categorical features
new_house = pd.get_dummies(new_house)

# 4. Match the training feature columns and their order
new_house = new_house.reindex(
    columns=X_train.columns,
    fill_value=0
)

# 5. Scale using the scaler fitted during training
new_house_scaled = scaler.transform(new_house)

# 6. Predict using your already-trained model
predicted_price = model.predict(new_house_scaled)

# 7. Display the prediction
print(f"Predicted House Price: ₹{predicted_price[0]:,.2f}")