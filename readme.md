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
- **Demographic Details**: Age, employment duration, home ownership.  
- **Financial Data**: Loan amount, funded amount, interest rate.  
- **Credit History**: Revolving balance, public records, total accounts.  
- **Loan Status**: Binary target variable indicating eligibility (0 or 1).  


## **Tools and Libraries**

- **Programming Language**: Python  
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

1. **Best Model**:  
   - **Random Forest** achieved the highest overall performance with robust handling of class imbalance.
   - Balanced Random Forest improved recall for the minority class.  

2. **Impact of Sampling**:
   - Oversampling improved recall but reduced precision due to synthetic data.  
   - Undersampling provided a better balance between precision and recall.

3. **Optimal Train-Test Split**:
   - A 60:40 split minimized overfitting while retaining sufficient training data.
   

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

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/loan-eligibility-prediction
   cd loan-eligibility-prediction
