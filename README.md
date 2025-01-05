# Handwritten Digit Classification Using Ensemble Learning

## Overview
This project focuses on classifying handwritten digits (0-9) from the **MNIST dataset** using an **ensemble of classifiers**. The ensemble combines three well-performing models: **Random Forest**, **Extra-Trees**, and **Support Vector Machine (SVM)**, using both **hard voting** and **soft voting** techniques. The final model leverages hyperparameter-tuned classifiers and achieves excellent performance on the test set.

## Skills Demonstrated
The following machine learning techniques were applied in this project:

1. **Data Preprocessing**:
   - Loaded the MNIST dataset and converted it to NumPy arrays.
   - Split the dataset into training, validation, and test sets.
   
2. **Model Training**:
   - Trained three classifiers independently:
     - Random Forest
     - Extra-Trees
     - SVM (RBF kernel)
   - Tuned hyperparameters for Random Forest and Extra-Trees using **RandomizedSearchCV**.
   
3. **Ensemble Learning**:
   - Combined classifiers using:
     - **Hard Voting**: Direct majority voting on predicted classes.
     - **Soft Voting**: Averaging predicted probabilities from classifiers.
   - Evaluated the ensemble performance on the validation and test sets.

## Results
- **Hyperparameter-tuned classifier accuracies**:
  - **Random Forest**: Validation accuracy ~ 97.09%
  - **Extra-Trees**: Validation accuracy ~ 97.36%
  - **SVM (RBF kernel)**: Validation accuracy ~ 98.52%
  
- **Ensemble accuracies**:
  - **Hard Voting Ensemble**: Validation accuracy ~ 97.40%
  - **Soft Voting Ensemble**: 
    - Validation accuracy ~ 98.41%
    - Test accuracy ~ 98.05%

## Conclusion
The **soft voting ensemble** achieved the best overall performance, with a final test accuracy of **98.05%**. This demonstrates the power of combining multiple models using ensemble techniques to improve robustness and generalization.

## Requirements
Ensure you have Python 3.7 or higher and the following libraries installed:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
