python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('Public_Transportation_Ridership_Data.csv')

# Data preprocessing
data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.to_period('M')

# Aggregating data by month
monthly_data = data.groupby('month').agg({
    'ridership': 'sum',
    'route': 'nunique'
}).reset_index()

# Visualize monthly ridership trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_data, x='month', y='ridership', marker='o')
plt.title('Monthly Public Transportation Ridership')
plt.xlabel('Month')
plt.ylabel('Ridership')
plt.grid()
plt.show()

# Generate a heatmap for busiest routes during peak hours
pivot_table = data.pivot_table(
    index='route', 
    columns='hour', 
    values='ridership', 
    aggfunc='sum', 
    fill_value=0
)

plt.figure(figsize=(15, 10))
sns.heatmap(pivot_table, cmap='coolwarm', annot=True, fmt='d')
plt.title('Heatmap of Ridership Across Routes and Hours')
plt.xlabel('Hour')
plt.ylabel('Route')
plt.show()

# Predictive analytics (example uses a simple linear regression model)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Feature engineering: Extract day of week and hour
data['day_of_week'] = data['date'].dt.dayofweek
features = data[['hour', 'day_of_week']]
labels = data['ridership']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict ridership for the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Example prediction for a specific day and hour
import numpy as np
predicted_ridership = model.predict(np.array([[10, 2]])) # Hour 10, Wednesday
print(f'Predicted Ridership at 10 AM on a Wednesday: {predicted_ridership[0]}')
