# Ex.No: 13 Learning – Use Supervised Learning  

### DATE : 10-10-2024  

### REGISTER NUMBER : 212221220025 

### AIM : 
To write a program to train the classifier for Sepsis Prediction. 


###  Algorithm :
1. Start the program.
2. Import required Python libraries, including NumPy, Pandas, Google Colab, Gradio, and various scikit-learn modules.
3.Load the sepsis dataset from a CSV file ('Paitients_Files_Train.csv') using Pandas.
4. Separate the target variable ('Outcome') from the input features and Scale the input features using the StandardScaler from scikit-learn.
5. Create a multi-layer perceptron (MLP) classifier model using scikit-learn's 'MLPClassifier'.
6. Train the model using the training data (x_train and y_train).
7. Define a function named 'sepsis' that takes input parameters for various features and Use the trained machine learning model to predict the outcome based on the input features.
8. Predict the accuracy.
9. Stop the program.

### Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pickle

```
```
# Load your dataset
data = pd.read_csv("Paitients_Files_Train.csv")

```
```
data.head()
```
```
data
```
```
# Extracting the target variable
y = data['Sepssis']

# Drop the target variable from the dataset for the features
X = data.drop(['Sepssis'], axis=1)

# Using LabelEncoder for the 'ID' column
label_encoder = LabelEncoder()
X['ID'] = label_encoder.fit_transform(X['ID'])
#X.drop('ID', axis=1, inplace=True)
```
```
data.info()
```
```
# Drop the original 'ID' column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    # Add more parameters to tune
}
```
```
# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```
```
# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```
```
# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Predict using the test set
predictions = best_model.predict(X_test)

```
```
# Model evaluation
print("Best Parameters:", best_params)
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Save the best model as a pickle file
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
```
```
#split data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x,y)
```
```
#scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)
```


### Output:

![image](https://github.com/user-attachments/assets/8cb86d3f-5189-4408-9e42-77f2accb5bf7)

![image](https://github.com/user-attachments/assets/70192c44-d223-4d42-8cc4-bac92b544c6d)

![image](https://github.com/user-attachments/assets/d341a8e9-e472-4fd6-b69e-da60738da8d3)

![image](https://github.com/user-attachments/assets/65791ba4-0ceb-472d-aa74-fefdcb980099)


### Result:
Thus the system was trained successfully and the prediction was carried out.
