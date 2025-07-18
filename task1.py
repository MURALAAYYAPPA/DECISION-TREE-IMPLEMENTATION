# Importing necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

print("Iris Dataset Preview:")
print(X.head())

# Split the dataset into training and testing sets (60% train, 40% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42,
    stratify=y  # maintain class distribution
)

# Initialize and train the Decision Tree model (limit depth to prevent overfitting)
dtree = DecisionTreeClassifier(max_depth=3, random_state=42)
dtree.fit(X_train, y_train)

# Predict on the test set
y_pred = dtree.predict(X_test)

# Evaluate the model
print("\n Model Performance:")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Display some sample predictions
print("\n Sample Predictions:")
comparison_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print(comparison_df.head(10))

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(
    dtree,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)
plt.title(" Decision Tree Visualization (max_depth=3)")
plt.tight_layout()
plt.show()
OUTPUT :
<img width="866" height="884" alt="Image" src="https://github.com/user-attachments/assets/edc47271-44c0-4f3f-92b8-fcde235d623a" />
<img width="1491" height="992" alt="Image" src="https://github.com/user-attachments/assets/269a507a-e849-44c2-a902-837bf375bb8f" />
