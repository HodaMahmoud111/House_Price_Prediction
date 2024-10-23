import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

train_data = pd.read_csv('C:/Users/Nvidia/Downloads/house-prices-advanced-regression-techniques (1)/train.csv')

print("Train Data:")
print(train_data.head(20))
print(train_data.info())
print(train_data.describe())

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
X = train_data[features]
y = train_data['SalePrice']

sns.histplot(train_data['SalePrice'], kde=True)
plt.title('Histogram of SalePrice')
plt.show()

sns.boxplot(x='GrLivArea', y='SalePrice', data=train_data)
plt.title('Boxplot of GrLivArea vs SalePrice')
plt.show()

sns.pairplot(train_data[features + ['SalePrice']])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

training_data_prediction = model.predict(X_train)
score_1 = metrics.r2_score(y_train, training_data_prediction)
score_2 = metrics.mean_absolute_error(y_train, training_data_prediction)

print('R Squared Error (Training):', score_1)
print('Mean Absolute Error (Training):', score_2)

test_data_prediction = model.predict(X_test)
score_3 = metrics.r2_score(y_test, test_data_prediction)
score_4 = metrics.mean_absolute_error(y_test, test_data_prediction)

print('R Squared Error (Test):', score_3)
print('Mean Absolute Error (Test):', score_4)

plt.scatter(y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price (Training Data)")
plt.show()

plt.scatter(y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price (Test Data)")
plt.show()

test_data = pd.read_csv('C:/Users/Nvidia/Downloads/house-prices-advanced-regression-techniques (1)/test.csv')
test_ids = test_data['Id']
test_data = test_data[features]

submission_test_data = test_data.copy()
submission_test_data['SalePrice'] = model.predict(test_data)

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': submission_test_data['SalePrice']
})
submission.to_csv('sample_submission.csv', index=False)

final_result = pd.read_csv('sample_submission.csv')
print(final_result)


