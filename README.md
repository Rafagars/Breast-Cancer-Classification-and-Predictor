# Breast Cancer Classification and Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project presents a machine learning-based classification system for diagnosing breast cancer using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. The primary goal is to accurately classify tumors as either benign or malignant based on their features.

An in-depth exploratory data analysis (EDA) was performed to understand feature distributions and correlations. The project explores advanced dimensionality reduction techniques (PCA, t-SNE, MDS) to visualize and process the data. Several supervised learning models were implemented and rigorously evaluated, including a Multilayer Perceptron (MLP), Support Vector Machine (SVM), and a Random Forest Classifier. The models were optimized using cross-validation and Grid Search to achieve high predictive performance, with a strong focus on maximizing recall to minimize false negatives, a critical requirement for medical diagnostic tools.

## Table of Contents

- [Key Features](#key-features)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Technologies Used](#technologies-used)
- [Setup and Usage](#setup-and-usage)
- [Results Summary](#results-summary)
- [License](#license)
- [Contact](#contact)

## Key Features

- **Comprehensive EDA:** In-depth analysis of feature distributions, correlations using heatmaps, and relationships between variables.
- **Advanced Dimensionality Reduction:** Implementation and evaluation of PCA, t-SNE, and MDS to visualize class separability in lower dimensions.
- **Multiple Model Implementation:** Systematic training and comparison of Multilayer Perceptron (MLP), Support Vector Machine (SVM), and Random Forest models.
- **Hyperparameter Optimization:** Use of `GridSearchCV` to find the optimal parameters for each model, enhancing predictive accuracy.
- **Robust Evaluation:** Focus on key metrics for medical diagnosis, including Confusion Matrices, Classification Reports (Precision, Recall, F1-Score), and ROC-AUC scores.
- **Feature Importance Analysis:** Extraction of feature importances from the Random Forest model to identify the most influential predictors of malignancy.

## Dataset

The project utilizes the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset, obtained from the UCI Machine Learning Repository.

- The dataset contains 569 instances and 32 attributes.
- The 'id' and an empty 'Unnamed: 32' column were removed during preprocessing.
- The target variable is 'diagnosis', which is binary (M = Malignant, B = Benign).
- The remaining 30 features are real-valued characteristics of the cell nuclei present in the digitized images of a fine needle aspirate (FNA) of a breast mass.

## Project Workflow

1.  **Data Loading and Cleaning:** The dataset is loaded, and unnecessary columns are dropped.
2.  **Exploratory Data Analysis (EDA):** The class distribution is visualized, and descriptive statistics are computed. A correlation heatmap and pairplots are generated to understand feature relationships.
3.  **Dimensionality Reduction:** PCA, t-SNE, and MDS are applied to the scaled feature set to visualize the data in 2D. The cumulative explained variance for PCA is analyzed to determine the optimal number of components.
4.  **Data Preprocessing:** The data is split into training (80%) and testing (20%) sets. `StandardScaler` is used to scale the features, which is essential for distance-based algorithms like SVM and MLP.
5.  **Model Training:** Three different classifiers are trained:
    -   Support Vector Machine (SVM)
    -   Multilayer Perceptron (MLP)
    -   Random Forest
6.  **Hyperparameter Tuning:** `GridSearchCV` is employed with 5-fold cross-validation to find the best hyperparameters for each of the three models.
7.  **Model Evaluation:** The tuned models are evaluated on the test set. Performance is measured using accuracy, a confusion matrix, a detailed classification report (with precision, recall, F1-score), and the ROC-AUC score.

## Technologies Used

- **Python 3.x**
- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical operations.
- **Matplotlib & Seaborn:** For data visualization.
- **Scikit-learn:** For machine learning tasks including:
  -   Data preprocessing (`StandardScaler`, `train_test_split`)
  -   Dimensionality reduction (`PCA`, `TSNE`, `MDS`)
  -   Modeling (`SVC`, `MLPClassifier`, `RandomForestClassifier`)
  -   Model selection and evaluation (`GridSearchCV`, `classification_report`, `roc_auc_score`)

## Setup and Usage

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Rafagars/Breast-Cancer-Classification-and-Predictor.git](https://github.com/Rafagars/Breast-Cancer-Classification-and-Predictor.git)
    cd Breast-Cancer-Classification-and-Predictor
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    (It's a good practice to create a `requirements.txt` file)
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

4.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook "Breast Cancer Predictor.ipynb"
    ```

## Results Summary

- All three optimized models achieved excellent performance, with AUC scores exceeding **0.99**, indicating outstanding classification capability.
- The **Support Vector Machine (SVM)** with a linear kernel and the **Random Forest** classifier showed slightly superior performance on the test set, both achieving around **96-98% accuracy**.
- Critically, the models achieved a high **recall** for the "malignant" class (often >0.95), successfully minimizing the number of dangerous false negatives.
- The feature importance analysis from the Random Forest model revealed that `area_worst`, `concave points_worst`, and `concave points_mean` were among the most significant predictors for the diagnosis.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.

## Contact

Rafael Garcia - [GitHub Profile](https://github.com/Rafagars) - [LinkedIn](https://www.linkedin.com/in/rafael-garcia-sanchez-2297b816a/) rafagar96@gmail.com
