# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset with relevant features and target variable
# Replace this with your actual dataset
# Assuming 'data' is your dataset with features and target variable

# Sample data (replace with your dataset)
data = pd.read_csv('your_dataset.csv')  # Update with the actual path to your dataset

# Separate features and target variable
X = data.drop('Approved', axis=1)  # Assuming 'Approved' is the target variable
y = data['Approved']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)

# Decision Tree model
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)

# Random Forest (ensemble) model
forest_model = RandomForestClassifier()
forest_model.fit(X_train, y_train)
forest_predictions = forest_model.predict(X_test)

# Ensemble method (voting classifier)
from sklearn.ensemble import VotingClassifier

ensemble_model = VotingClassifier(estimators=[
    ('logistic', logistic_model),
    ('tree', tree_model),
    ('forest', forest_model)
], voting='hard')

ensemble_model.fit(X_train, y_train)
ensemble_predictions = ensemble_model.predict(X_test)

# Evaluate the models
print("Logistic Regression Accuracy:", accuracy_score(y_test, logistic_predictions))
print("Decision Tree Accuracy:", accuracy_score(y_test, tree_predictions))
print("Random Forest Accuracy:", accuracy_score(y_test, forest_predictions))
print("Ensemble Model Accuracy:", accuracy_score(y_test, ensemble_predictions))

# You can also print classification reports for more detailed metrics
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, logistic_predictions))
print("\nDecision Tree Classification Report:\n", classification_report(y_test, tree_predictions))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, forest_predictions))
print("\nEnsemble Model Classification Report:\n", classification_report(y_test, ensemble_predictions))
