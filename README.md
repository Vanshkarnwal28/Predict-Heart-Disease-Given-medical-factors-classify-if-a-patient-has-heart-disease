# Predict-Heart-Disease-Given-medical-factors-classify-if-a-patient-has-heart-disease 
Project Overview

This project focuses on predicting heart disease using a machine learning model (Random Forest Classifier). The dataset used consists of patient medical information, and the task is to classify whether a patient has heart disease or not. The model is evaluated using accuracy, precision, recall, and a confusion matrix heatmap.

Problem Statement

To generate heatmaps of confusion matrices and calculate evaluation metrics such as accuracy, precision, and recall for classification problems. For non-classification problems, segmentation and clustering techniques are to be applied.

Dataset

File Name: 4. Predict Heart Disease.csv

Source: UCI Machine Learning Repository

Attributes: Includes features like age, sex, chest pain type, resting blood pressure, cholesterol, etc., and a target variable indicating the presence of heart disease.

Technologies Used

Python 3.x

pandas

seaborn

matplotlib

scikit-learn

Steps Performed

Data Loading: CSV file is loaded using pandas.

Data Preprocessing: Checked for nulls and dropped target variable from features.

Model Training: Random Forest Classifier is used for training.

Model Evaluation:

Confusion Matrix

Accuracy

Precision

Recall

Visualization: Confusion Matrix heatmap is plotted using seaborn.

Code Highlights

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

These libraries are used to build and evaluate the model.

Output Example

Accuracy: 0.85

Precision: 0.86

Recall: 0.88

Confusion matrix heatmap is generated and visualized using matplotlib and seaborn.

How to Run

Clone the repository or download the script.

Place the CSV dataset in the same directory.

Run the script in a Python environment (e.g., Jupyter Notebook).

References

UCI Heart Disease Dataset

scikit-learn Documentation
![image](https://github.com/user-attachments/assets/cae9f666-3435-41d5-854b-8e1a82cd99e0)

