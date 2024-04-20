# Machine Learning Project

In this project, the aim is to develop using only Numpy tree models such as decision trees, bagging, random forests, gradient boosting and others 


## Modules

- **base**:
    - Contains `BinaryTreeNode` which creates a one shot node (parent node) and two connected children nodes (left node and right node)
    - Contains the `DecisionTreeClassifier` model which splits the dataset iteratively based on given criterion.
      It includes model fittig, prediction and visualisation of tree 

- **metrics**: Stores various metrics used for:

  - Regression:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
  - Classification:
    - Accuracy
    - Recall
    - Precision
    - F1 Score
  - Others:
    - Confusion Matrix
    - Cross Validation

- **err_handl**: Manages errors within the modules.

- **dgp**: Generates data with different characteristics used for linear regression and logistic regression



## Testing

All the work is thoroughly tested and summarized in the notebook folder. These notebooks execute, test, and evaluate tree models.





## Versions

- Python Version: 3.11.4
- Numpy Version : 1.26.4
- Pandas Version : 2.2.1
