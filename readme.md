# Loan Eligibility Prediction Using Machine Learning

This project explores the development of an optimized machine learning model for predicting loan eligibility. Using a dataset containing 67,463 rows and 35 attributes, we evaluated multiple machine learning techniques to improve accuracy, interpretability, and fairness in credit risk assessments. The research provides actionable insights for deploying trustworthy AI in the financial services domain.


## **Problem Statement**

Manual lending decisions are inconsistent, subjective, and prone to delays, leading to suboptimal credit risk assessment. Traditional methods, such as regression models, struggle with adapting to diverse data and capturing complex interactions. Machine learning provides scalable, data-driven solutions to accurately predict loan eligibility while maintaining transparency and accountability.


## **Objectives**

- Develop interpretable machine learning models for loan eligibility prediction.
- Handle class imbalance effectively using techniques like undersampling and SMOTE.
- Evaluate models on metrics such as accuracy, precision, recall, F1 Score, and AUC-ROC.
- Provide recommendations for deploying responsible AI in financial services.


## **Research Questions**

1. What machine learning models and hyperparameters optimize predictive performance for credit risk modeling?  
2. What are the biases and limitations in the models’ predictions, and how can they be mitigated through responsible AI?  
3. How do the model’s predictions align with domain expert assessments for real-world deployment?


## **Dataset**

### **Source**
- **Platform**: Kaggle  
- **URL**: [Loan Default Prediction Dataset](https://www.kaggle.com/datasets/hemanthsai7/loandefault)  
- **Structure**: 67,463 rows and 35 attributes  

### **Key Features**

| **Feature Name**           | **Description**                          | **Type**         |
|-----------------------------|------------------------------------------|------------------|
| `Loan_ID`                  | Unique identifier for the loan           | Categorical      |
| `Gender`                   | Gender of the applicant                  | Categorical      |
| `Married`                  | Applicant's marital status               | Categorical      |
| `Dependents`               | Number of dependents                     | Ordinal          |
| `Education`                | Education level of the applicant         | Categorical      |
| `Self_Employed`            | Indicates if the applicant is self-employed | Categorical    |
| `ApplicantIncome`          | Monthly income of the applicant          | Continuous       |
| `CoapplicantIncome`        | Monthly income of the co-applicant       | Continuous       |
| `LoanAmount`               | Amount of the loan requested             | Continuous       |
| `Loan_Amount_Term`         | Term (duration) of the loan (in months)  | Continuous       |
| `Credit_History`           | Record of previous credit history        | Ordinal          |
| `Property_Area`            | Type of area where property is located   | Categorical      |
| `Loan_Status`              | Status of loan approval (Y/N)            | Categorical      |

 


## **Tools and Libraries**

-  Python  
- **Libraries**:
  - Data Processing: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`  
  - Machine Learning: `scikit-learn`, `xgboost`, `lightgbm`, `catboost`  
  - Class Imbalance Management: `imblearn`  
  - Model Explainability: `shap`, `lime`  


## **Project Workflow**

### **1. Data Understanding and Preparation**
- Performed exploratory data analysis to identify class imbalance and variable distributions.  
- Cleaned data by removing irrelevant and redundant features.  
- Applied one-hot encoding for categorical variables.  
- Handled class imbalance using SMOTE and undersampling techniques.

### **2. Modeling**
- Evaluated six machine learning models:
  - Random Forest
  - Balanced Random Forest
  - AdaBoost
  - Gradient Boosting
  - Logistic Regression
  - Decision Tree
- Used grid search and cross-validation for hyperparameter tuning.

### **3. Evaluation**
- Metrics:
  - **Accuracy**: Overall correctness of predictions.  
  - **Precision**: Fraction of correctly predicted positives.  
  - **Recall**: Fraction of actual positives identified.  
  - **F1 Score**: Harmonic mean of precision and recall.  
  - **AUC-ROC**: Model’s ability to differentiate between classes.  
- Confusion matrices and ROC curves visualized model performance.  

### **4. Explainability**
- Global Interpretability: Feature importance, Partial Dependence Plots.  
- Local Interpretability: SHAP and LIME for instance-level explanations.


## **Key Findings**

1. **Top-Performing Models**:  
   - The **Random Forest Classifier** emerged as the most effective model, achieving the highest overall performance across metrics such as accuracy, precision, recall, F1 score, and AUC-ROC.  
   - **Balanced Random Forest** effectively addressed class imbalance, significantly improving recall for the minority class (loan defaults) while maintaining robust overall performance.

2. **Effect of Sampling Techniques**:  
   - **Oversampling** (e.g., using SMOTE) increased recall by exposing the model to more instances of the minority class. However, it introduced synthetic data, leading to a slight decrease in precision.  
   - **Undersampling** balanced class distribution by reducing the majority class, resulting in a better trade-off between precision and recall while preserving data quality. A 60:40 undersampling ratio provided the best results.

3. **Optimal Train-Test Split Ratio**:  
   - A **60:40 split** was identified as the optimal ratio, striking a balance between retaining sufficient training data and minimizing overfitting. Lower train ratios (e.g., 55:45) further reduced overfitting but slightly compromised generalization on unseen data.

4. **Model Explainability**:  
   - Feature importance analysis revealed critical predictors of loan defaults, such as `Credit_History`, `LoanAmount`, `ApplicantIncome`, and `Property_Area`.  
   - SHAP and LIME provided local and global interpretability, enhancing trust and accountability in the models.

5. **Business Implications**:  
   - Models like Random Forest can transform the loan eligibility process by providing automated, data-driven predictions, minimizing default risk, and improving financial inclusion.
   - Balanced sampling strategies and explainable AI frameworks ensure ethical, transparent, and fair decision-making.

   

## **Results Summary**

| Model                  | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|------------------------|----------|-----------|--------|----------|---------|
| Random Forest          | 0.9097   | 0.90      | 0.91   | 0.90     | 0.92    |
| Balanced Random Forest | 0.5410   | 0.50      | 0.55   | 0.52     | 0.51    |
| AdaBoost               | 0.9090   | 0.89      | 0.90   | 0.89     | 0.91    |
| Gradient Boosting      | 0.9086   | 0.88      | 0.90   | 0.89     | 0.91    |
| Logistic Regression    | 0.5191   | 0.48      | 0.50   | 0.49     | 0.50    |


## **Conclusion**

This research demonstrates the potential of machine learning for loan eligibility prediction. By integrating sampling techniques, ensemble methods, and explainability tools, we developed a robust and interpretable model. The findings contribute to the growing body of research on responsible AI for financial decision-making.


## **How to Run the Project**

1. **Clone the Repository**:
   - Open your terminal or command prompt and run the following commands:
     ```bash
     git clone https://github.com/your-username/loan-eligibility-prediction
     cd loan-eligibility-prediction
     ```

2. **Install Required Libraries**:
   - Ensure you have Python 3.7 or later installed on your system.
   - Manually install the required libraries listed in the project (if not already installed):
     ```bash
     pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn shap
     ```

3. **Prepare the Dataset**:
   - Verify the dataset (e.g., `train.csv`) is included in the repository.
   - If missing, download it from the source and place it in the `data/` directory.

4. **Run the Project Notebook**:
   - Open the Jupyter Notebook (e.g., `Loan_Prediction.ipynb`) in Jupyter Notebook, VS Code, or any compatible environment.
   - Execute the cells step-by-step:
     - Handle missing values, feature encoding, and other cleaning steps.
     - Train models like Random Forest, Gradient Boosting, etc.
     - Evaluate the models using metrics like accuracy, precision, recall, and F1-score.

5. **Review Outputs**:
   - Check the outputs in the notebook:
     - Metrics and graphs for model performance.
     - SHAP or other explainability visualizations.


### **Important Notes**
- Ensure the dataset structure matches the notebook's expectations.
- For questions, refer to the comments and markdown sections within the notebook.

