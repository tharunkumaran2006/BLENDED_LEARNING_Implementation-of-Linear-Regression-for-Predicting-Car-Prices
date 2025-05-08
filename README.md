# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import Libraries
Import necessary Python libraries such as pandas, numpy, matplotlib, seaborn, sklearn, and statsmodels.

#### 2.Load Dataset
Read the dataset CarPrice_Assignment.csv using pandas.read_csv() and assign predictor variables (features) and target variable (price).

#### 3.Split Dataset
Use train_test_split() to divide the dataset into training and testing sets.

#### 4.Data Preprocessing
Normalize the features using StandardScaler to standardize the input data for better model performance.

#### 5.Model Training
Train the linear regression model on the scaled training data using LinearRegression().fit().

#### 6.Model Prediction
Use the trained model to predict prices for the test data.

#### 7.Evaluate Model
Calculate and print the model’s coefficients, intercept, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score to evaluate model performance.

#### 8.Linearity Check
Plot the actual vs predicted values to visually inspect if the relationship is linear.

#### 9.Check Independence of Errors
Compute the Durbin-Watson statistic using statsmodels to test for autocorrelation in residuals.

#### 10.Check Homoscedasticity
Plot residuals against predicted values to ensure that the residuals have constant variance.

#### 11.Check Normality of Residuals
Plot histogram and Q-Q plot of residuals to check if they follow a normal distribution.

## Program:
### Program to implement linear regression model for predicting car prices and test assumptions.
### Developed by: THARUN V K
### RegisterNumber:  212223230231
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
df=pd.read_csv('CarPrice_Assignment.csv')
x=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
model=LinearRegression()
model.fit(x_train_scaled,y_train)
y_pred=model.predict(x_test_scaled)
print("="*50)
print("MODEL COEFFICIENTS:")
for feature,coef in zip(x.columns,model.coef_):
    print(f"{feature:>12}: {coef:>10.2f}")
print(f"{'Intercept':>12}:{model.intercept_:>10.2f}")
print("\n MODEL PERFORMANCE:")
print(f"{'MSE':>12}: {mean_squared_error(y_test,y_pred):>10.2f}")
print(f"{'RMSE':>12}:{np.sqrt(mean_squared_error(y_test,y_pred)):>10.2f}")
print(f"{'R-squares':>12}:{r2_score(y_test,y_pred):10.2f}")
print("="*50)

# 1. Linearity check
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()

# 2. Independence (Durbin-Watson)
residuals = y_test - y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}",
      "\n(Values close to 2 indicate no autocorrelation)")

# 3. Homoscedasticity
plt.figure(figsize=(10, 5))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()
# 4. Normality of residuals
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(residuals, kde=True, ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45', fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()
```

## Output:
#### MODEL COEFFICIENTS AND MODEL PERFORMANCE:
![Screenshot 2025-04-23 190236](https://github.com/user-attachments/assets/f00cf887-bf33-4878-ba04-6bbacd120f75)

![Screenshot 2025-04-23 190245](https://github.com/user-attachments/assets/6e7182a6-b28f-429e-8097-e3a2a05b160f)

#### LINEARITY CHECK:
![Screenshot 2025-04-23 190303](https://github.com/user-attachments/assets/9ba28ad7-920b-4131-988c-73f18caa76ef)

#### INDEPENDENCE (DURBIN-WATSON):
![Screenshot 2025-04-23 190311](https://github.com/user-attachments/assets/7c98dc37-24e7-4d9a-9026-8024f58c3645)

#### HOMOSCEDASTICITY:
![Screenshot 2025-04-23 190319](https://github.com/user-attachments/assets/1b8430b6-ed9d-438a-984f-43df771836cd)


#### NORMALITY OF RESIDUALS:
![Screenshot 2025-04-23 190336](https://github.com/user-attachments/assets/8e9445d1-d893-47e5-892a-c47272fc7e63)



## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
