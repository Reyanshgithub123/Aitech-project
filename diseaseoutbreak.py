import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('disease_data.csv')

# Display first few rows
print(df.head())

# Data Preprocessing
features = ['temperature', 'humidity', 'population_density', 'healthcare_access', 'infected_neighbors']
target = 'outbreak_risk'
X = df[features]
y = df[target]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=model.feature_importances_, y=features)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance in Disease Outbreak Prediction')
plt.show()

# Predict on new data
def predict_outbreak(data):
    prediction = model.predict([data])
    return 'High Risk' if prediction[0] == 1 else 'Low Risk'

# Example Prediction
sample_data = [30, 80, 500, 4, 20]  # Sample feature values
print('Predicted Outbreak Risk:', predict_outbreak(sample_data))
