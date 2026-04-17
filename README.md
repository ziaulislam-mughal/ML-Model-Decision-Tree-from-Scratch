<h1 style="text-align: center; font-family: 'Times New Roman', Times, serif; font-weight: bold;">🌳 Decision Tree from Scratch</h1>

### **📌 Overview**

This module is responsible for evaluating the performance of the custom **Decision Tree model** built from scratch. It loads the dataset, splits it into training and testing sets, trains the decision tree model, makes predictions, and calculates evaluation metrics to measure the model's performance.

The evaluation metrics help determine how accurately the model predicts the correct class labels and how well it generalizes to unseen data.

### **📚 Libraries Used**


```python
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
```


#### **Pandas**

`pandas` is used for handling and manipulating structured datasets such as CSV files or tabular data. It allows easy loading, cleaning, and preprocessing of the dataset.

#### **NumPy**

`numpy` is used for numerical operations and efficient array manipulation. It helps perform mathematical operations on datasets.

#### **Scikit-learn Metrics**

The sklearn.metrics module is used to evaluate the performance of the model. It provides built-in functions to calculate important evaluation metrics.

The following `metrics` are used:

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- Confusion Matrix
- Train Test Split

`train_test_split` from sklearn.model_selection is used to divide the dataset into two parts:

- **Training Data (used to train the model)**
- **Testing Data (used to evaluate the model)**

This helps ensure that the model is tested on unseen data.

## **Custom Decision Tree Functions**

It supports both **categorical** and continuous features using **Entropy** & **Information Gain.**

### **📊 1. Entropy**

Entropy measures the impurity or randomness in the dataset.

`Formula Idea:`
High entropy → mixed classes
Low entropy → pure class

```python

def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    # formula: -Σ p * log2(p)
    return -np.sum(probs * np.log2(probs + 1e-9))

    
```
**🔍 Explanation:**

- np.unique(y) → finds classes and their counts <br>
- probability = count / total <br>
- final entropy = sum of -p log2(p) <br>


---
### **📈 2. Information Gain**

Information Gain tells us how good a split is.<br>

`Idea:`

Higher gain = better feature for splitting

```python
def information_gain(y, x, threshold=None):
    parent_entropy = entropy(y)
    n = len(y)

```

#### **🔹 Case 1: Continuous Feature Split**
```python 
    if threshold is not None:
        left  = y[x <= threshold]
        right = y[x > threshold]

        if len(left) == 0 or len(right) == 0:
            return 0

        weighted = (len(left)/n)*entropy(left) + (len(right)/n)*entropy(right)
        
```

 **Explanation**

- Split data into left and right groups
- Compute weighted entropy
- If one side is empty → invalid split

### **🔹 Case 2: Categorical Feature Split**
```python
    else:
        weighted = sum(
            (len(y[x == v])/n) * entropy(y[x == v])
            for v in np.unique(x)
        )
``` 

**Explanation:**
- Split based on each category value
- Compute entropy for each category
- Take weighted average

####  **📌 Final Gain Calculation**
``` python
    return parent_entropy - weighted 
```

👉 Information Gain = Parent Entropy − Child Entropy

---
### **🎯 3. Best Threshold Selection**

For continuous features, we test multiple split points.
```python
def best_threshold(y, x):
    sorted_vals = np.sort(np.unique(x))
    thresholds  = (sorted_vals[:-1] + sorted_vals[1:]) / 2
    gains = [information_gain(y, x, t) for t in thresholds]
    return thresholds[np.argmax(gains)], max(gains)
```
**Explanation:**

- Find midpoints between values
- Try each as split point
- Select threshold with maximum information gain

---
### **🌿 4. Tree Node Structure**

Each node represents:

- a decision (feature + threshold)
- or a final prediction (leaf node)
```python
class Node:
    def __init__(self, feature=None, threshold=None, value=None):
        self.feature   = feature
        self.threshold = threshold
        self.value     = value
        self.children  = {}
        self.left      = None
        self.right     = None

```

**Types of Nodes:**
- 🌱 Leaf Node → stores final class (value)
- 🌿 Decision Node → splits data further

---

### **🧠 5. Majority Class Function**

Used when stopping tree growth.
```python
def majority(y):
    vals, counts = np.unique(y, return_counts=True)
    return vals[np.argmax(counts)]
```
**Explanation:**
  Returns the most frequent class.

---
### **🌳 6. Building the Tree**

Core recursive function that builds the decision tree.
```py
def build_tree(X, y, depth=0, max_depth=10, min_samples=2):
```
##### **🚫 Base Conditions (Stopping Rules)**
```python
    if len(np.unique(y)) == 1:
        return Node(value=y[0])

    if depth >= max_depth or len(y) < min_samples:
        return Node(value=majority(y))

```
**Meaning:**
- Pure class → stop
- Max depth reached → stop
- Too few samples → stop


#### **🔍 Find Best Feature**
```python
    best_gain, best_feat, best_thresh = -1, None, None

    for col in X.columns:
        x = X[col].values

        if X[col].dtype in [np.float64, np.int64]:
            thresh, gain = best_threshold(y, x)
        else:
            gain = information_gain(y, x)
            thresh = None
```

**Explanation:**
- Try every feature
- Compute information gain
- Select best one

##### **🌿 Split the Node**
```python
    node = Node(feature=best_feat, threshold=best_thresh)
    x    = X[best_feat].values
```

##### **🔹 Continuous Split**
```python
    if best_thresh is not None:
        node.left  = build_tree(X[x <= best_thresh], y[x <= best_thresh])
        node.right = build_tree(X[x > best_thresh], y[x > best_thresh])
```

##### **🔹 Categorical Split**
```py
    else:
        for val in np.unique(x):
            mask = x == val
            node.children[val] = build_tree(
                X[mask].drop(columns=[best_feat]),
                y[mask]
            )
```
---
### **🔮 7. Prediction Function**
##### **Predict One Sample**
```python

def predict_one(node, row):
    if node.value is not None:
        return node.value

```

##### **Continuous Decision**
```python
    if node.threshold is not None:
        child = node.left if row[node.feature] <= node.threshold else node.right
```
##### **Categorical Decision**
```python
    else:
        child = node.children.get(
            row[node.feature],
            next(iter(node.children.values()))
        )
```
##### **Recursive Call**
```py
    return predict_one(child, row)
```
---
### **📦 Predict Full Dataset**
```python
def predict(tree, X):
    return np.array([predict_one(tree, row) for _, row in X.iterrows()])
```
---
### **🎯 Summary**

This implementation builds a Decision Tree Classifier from scratch using:

- 📌 Entropy → impurity measurement
- 📌 Information Gain → best split selection
- 📌 Recursive tree building
- 📌 Support for categorical + continuous data
- 📌 Custom prediction function


```python

from sklearn.model_selection import train_test_split
from decision_tree import build_tree, predict 
```



The project uses a custom decision tree implementation instead of using built-in machine learning models.

The following functions are imported from the **decision tree module.**

`build_tree()`

This function constructs the decision tree using the training dataset. It recursively splits the dataset based on the best feature until a stopping condition is met.

The output of this function is a tree structure representing the decision rules learned from the dataset.

`predict()`

This function is used to generate predictions from the trained decision tree. It takes the trained tree and input features as arguments and returns the predicted class label.

### **Workflow of the project**

The project follows these steps:

**1.** Load the dataset using pandas.<br>
**2.** Separate features and target labels.<br>
**3.** Split the dataset into training and testing sets.<br>
**4.** Train the decision tree using the training data.<br>
**5.** Generate predictions using the test data.<br>
**6.** Compare predictions with actual values.<br>
**7.** Calculate evaluation metrics.<br>
**8.** Display the results.<br>

### **1. Load Dataset**


```python
df = pd.read_csv("tennis.csv")
```

This line loads the dataset `tennis.csv` using the pandas library and stores it in a DataFrame named **df**.
A DataFrame is a table-like structure used for storing and manipulating data.

#### **Check Missing Values**


```python
# Missing data in dataset 
df.isnull().sum()
```




    outlook     0
    temp        0
    humidity    0
    windy       0
    play        0
    dtype: int64





This command checks the dataset for **missing (null) values** in each column.
It returns the total number of missing entries present in every feature.

### **Check Duplicate Records**




```python
df.duplicated().sum()
```




    0



This line identifies **duplicate** **rows** in the dataset.
It counts how many records appear more than once.

### **Dataset Information**


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14 entries, 0 to 13
    Data columns (total 5 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   outlook   14 non-null     object
     1   temp      14 non-null     object
     2   humidity  14 non-null     object
     3   windy     14 non-null     bool  
     4   play      14 non-null     object
    dtypes: bool(1), object(4)
    memory usage: 594.0+ bytes
    

This function provides a **summary** of the **dataset**, including column names, data types, and the number of non-null values.
It helps understand the structure and quality of the dataset.

## **2. Separate features and target labels.**

### **Define Features (Input)**


```python
# Features (input)
X = df.drop(columns=["play"])
```


This line creates the feature dataset (X) by removing the target column **"play"** from the DataFrame.
The remaining columns are used as **input variables** for training the machine learning model.

### **Define Target (output)**


```python
# Target (output)
y = df["play"].values
```

This line extracts the **target variable (y)** from the dataset, which is the **"play"**column.
It represents the **label or output** **that the model will predict**, and **.values** converts it into a NumPy array for easier processing.

## **3. Split the dataset into training and testing sets.<br>**


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

This line splits the dataset into **training and testing sets** using `train_test_split`.
80% of the data is used to train the model and 20% is reserved for testing its performance on unseen data. The `random_state=42` ensures the split is reproducible every time the code runs.

## **4. Train the Decision Tree Model**


```python
tree = build_tree(X_train, y_train, max_depth=10)
```

This line trains the **Decision Tree model** using the training dataset (`X_train` and `y_train`).
The `build_tree()` function constructs the tree by recursively splitting the data, while **max_depth=10** limits the tree depth to prevent overfitting.

## **5. Generate Predictions**


```python
preds = predict(tree, X_test)
```


This line uses the **trained decision tree (tree)** to make predictions on the **test dataset (X_test).**
The `predict()` function traverses the tree for each test sample and returns the predicted class labels, which are stored in preds.

## **6. Compare Predicted and Actual Values**





```python
for i in range(len(preds)):
    print(f"Actual: {y_test[i]}  |  Predicted: {preds[i]}")
```

    Actual: yes  |  Predicted: yes
    Actual: yes  |  Predicted: yes
    Actual: no  |  Predicted: no
    

This loop iterates through all test samples and compares the **actual values (y_test)** with the **predicted values (preds).**

For each index, it prints both values side-by-side, allowing you to easily see where the model predicted correctly or incorrectly.

## **7. Calculate Evaluation metrics.**

#### **Calculate Model Accuracy**


```python
accuracy_score_result = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy_score_result:.2f}")


```

    Accuracy: 1.00
    

The **accuracy_score()** function returns the percentage of correct predictions, and the result is printed with two decimal places for better readability.

#### **Calculate Precision Score**


```python
precision_score = precision_score(y_test, preds, average="weighted")
print(f"Precision: {precision_score:.2f}")


```

    Precision: 1.00
    

This code calculates the precision of the model, which measures how many of the predicted positive results are actually correct.

The average="weighted" parameter is used for multi-class classification, where each class is weighted by its support (number of true instances). The result is then printed with two decimal places for clarity.

#### **Recall Score**


```python
recall_score = recall_score(y_test, preds, average="weighted")
print(f"Recall: {recall_score:.2f}")


```

    Recall: 1.00
    

This code calculates the recall of the model, which measures how many actual positive cases were correctly identified by the model.

The average="weighted" parameter is used for multi-class classification, where each class contributes proportionally based on its number of samples. The result is printed with two decimal places for readability.

####  **Calculate F1 Score**





```python
f1_score = f1_score(y_test, preds, average="weighted")
print(f"F1 Score: {f1_score:.2f}")


```

    F1 Score: 1.00
    

This code calculates the F1 Score, which is the harmonic mean of precision and recall.

It provides a balanced measure of model performance, especially when the dataset has class imbalance. The average="weighted" parameter ensures each class is considered based on its frequency, and the result is printed with two decimal places.

#### **Confusion Matrix + TP, TN, FP, FN**


```python
from sklearn.metrics import confusion_matrix

# Step 1: Generate confusion matrix
cm = confusion_matrix(y_test, preds)

print("Confusion Matrix:")
print(cm)


# Step 2: Extract values (for binary classification)
# cm structure:
# [[TN, FP],
#  [FN, TP]]

TN, FP, FN, TP = cm.ravel()

print("\n--- Detailed Metrics ---")
print(f"True Positive (TP): {TP}")   # Correct positive predictions
print(f"True Negative (TN): {TN}")   # Correct negative predictions
print(f"False Positive (FP): {FP}")  # Wrong positive predictions
print(f"False Negative (FN): {FN}")  # Missed positive predictions

```

    Confusion Matrix:
    [[1 0]
     [0 2]]
    
    --- Detailed Metrics ---
    True Positive (TP): 2
    True Negative (TN): 1
    False Positive (FP): 0
    False Negative (FN): 0
    

The confusion matrix is used to evaluate classification models by comparing actual values vs predicted values.
It helps measure how well the model is performing on different classes.

#### **Outcomes of Confusion Matrix**
- True Positive (TP):
Model correctly predicts the positive class.
- True Negative (TN):
Model correctly predicts the negative class.
- False Positive (FP):
Model incorrectly predicts positive when it is actually negative (Type I error).
- False Negative (FN):
Model incorrectly predicts negative when it is actually positive (Type II error).

## 📌 Conclusion

- A Decision Tree was successfully built from scratch.
- The model was trained using entropy and information gain.
- Evaluation metrics show model performance on unseen data.
- This project improves understanding of machine learning internals without using prebuilt models.


```python

```
