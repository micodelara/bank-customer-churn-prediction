# Bank Customer Churn Prediction Using Supervised Machine Learning

**Customer Analytics | Predictive Modeling | Explainable AI | Python | Business Analytics**

## Project Overview

Customer churn is a major challenge for financial institutions because losing existing customers can reduce revenue, increase acquisition costs, and weaken long-term customer relationships.

This project applies **supervised machine learning** to predict which bank customers are at risk of leaving and identify the factors contributing to their churn probability.

Using a dataset of 10,000 bank customers, the study compares six classification algorithms, optimizes model performance using the recall-weighted F2-score, and applies **SHapley Additive exPlanations (SHAP)** to interpret model predictions.

The project also translates analytical findings into actionable customer retention strategies.

## 1. Business Problem

Banks need to identify customers who are likely to discontinue their relationship with the institution before they actually leave.

However, customer churn is influenced by multiple characteristics, including age, account balances, product holdings, engagement levels, and other customer attributes.

Traditional retention approaches may treat all customers similarly, resulting in inefficient allocation of retention resources.

**Central Business Question:**

How can supervised machine learning identify customers at high risk of churning, explain the factors influencing their predicted risk, and support targeted customer retention interventions?

### Project Objectives

1. Identify customers at high risk of leaving the bank.
2. Explore customer characteristics associated with churn.
3. Develop and compare supervised classification models.
4. Prioritize recall using the F2-score to reduce missed churners.
5. Optimize the selected model through hyperparameter tuning.
6. Identify influential predictors using SHAP.
7. Translate model findings into targeted customer retention strategies.

## 2. Dataset

**Dataset:** Bank Customer Churn Prediction Dataset

**Source:** Kaggle

**Dataset Link:** https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset

### Dataset Characteristics

| Characteristic | Description |
|---|---|
| Number of observations | 10,000 customers |
| Original variables | 14 |
| Target variable | `Exited` |
| Churn rate | 20.37% |
| Missing values | None |
| Duplicate records | None |
| Training/test split | 70% / 30% |
| Modeling task | Binary classification |

### Target Variable

`Exited` indicates whether a customer discontinued their relationship with the bank.

- **0:** Customer stayed
- **1:** Customer exited

### Predictor Variables

| Variable | Description |
|---|---|
| `CreditScore` | Customer credit score |
| `Geography` | Customer's geographic location |
| `Age` | Customer age |
| `Tenure` | Years with the bank |
| `Balance` | Customer account balance |
| `NumOfProducts` | Number of bank products held |
| `HasCrCard` | Credit card ownership indicator |
| `IsActiveMember` | Customer activity indicator |
| `EstimatedSalary` | Estimated customer salary |

The identifier fields `RowNumber`, `CustomerId`, and `Surname` were excluded from modeling. `Gender` was also excluded based on the project's ethical considerations.

## 3. Analytical Methodology

### Step 1: Data Preparation

The preprocessing workflow included:

- Importing the dataset using `kagglehub`.
- Inspecting data types and summary statistics.
- Checking missing values and duplicate records.
- Removing identifier fields and gender.
- Applying one-hot encoding to geography.
- Examining alternative age transformations.
- Splitting the dataset into training and testing subsets.
- Standardizing predictors for algorithms requiring feature scaling.

The dataset was divided into:

| Dataset | Observations |
|---|---:|
| Training | 7,000 |
| Testing | 3,000 |

Stratified sampling was used to preserve the approximate churn distribution across training and testing datasets.

### Step 2: Age Transformation

Age was positively skewed in the original dataset.

Three representations were compared:

- Original age
- Log-transformed age
- Yeo-Johnson transformation

| Transformation | Skewness |
|---|---:|
| Original | 1.011 |
| Log transformation | 0.203 |
| Yeo-Johnson | -0.002 |

The Yeo-Johnson transformation was selected because it substantially reduced skewness.

The transformation was fitted using the training data and subsequently applied to the testing data to avoid information leakage.

### Step 3: Exploratory Data Analysis

The exploratory analysis included:

- Histograms and density plots
- Box plots
- Churn rates by customer category
- Pair plots
- Correlation analysis
- Product-level churn comparisons

### Key EDA Findings

**Customer Age**

Customers who exited tended to be older than customers who remained with the bank.

**Customer Engagement**

Inactive customers had a churn rate of approximately 26.9%, compared with 14.3% among active customers.

**Number of Products**

The relationship between product holdings and churn was nonlinear.

| Number of Products | Churn Rate |
|---|---:|
| 1 | 27.7% |
| 2 | 7.6% |
| 3 | 82.7% |
| 4 | 100% |

Customers holding three or four products represented a relatively small proportion of the dataset, so these unusually high churn rates should be interpreted cautiously.

**Geographic Differences**

Germany had a churn rate of approximately 32.4%, compared with around 16% in France and Spain.

**Other Variables**

Credit score, tenure, estimated salary, and credit card ownership showed weaker standalone relationships with churn during exploratory analysis.

## 4. Supervised Machine Learning Models

Six classification algorithms were evaluated.

### Logistic Regression

A baseline linear classification approach was developed using standardized predictors and balanced class weights.

### Random Forest

An ensemble of decision trees was used to capture nonlinear relationships and interactions among customer characteristics.

### Gradient Boosting

A sequential tree-based ensemble algorithm was used to improve classification performance, with sample weights applied to address class imbalance.

### Support Vector Machine

Linear, polynomial, and radial basis function kernels were explored using different class-weight configurations.

### Gaussian Naive Bayes

A probabilistic classification model was developed and tuned using the variance smoothing parameter.

### K-Nearest Neighbors

A distance-based classification model was implemented using standardized features and hyperparameter optimization.

## 5. Model Evaluation

### Primary Selection Metric: F2-Score

The study prioritized the **F2-score** because missing a customer who will actually churn may be more costly than incorrectly identifying a customer as high-risk.

Compared with F1-score, F2-score places greater emphasis on recall.

This aligns model selection with the business objective of identifying as many potential churners as reasonably possible.

### Initial Model Comparison

| Model | Accuracy | Recall | F2-Score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.686 | 0.718 | 0.601 | 0.765 |
| Random Forest | 0.822 | 0.669 | 0.642 | 0.854 |
| **Gradient Boosting** | **0.805** | **0.722** | **0.668** | **0.857** |
| SVM (RBF, Balanced) | 0.792 | 0.720 | 0.659 | 0.843 |
| Naive Bayes | 0.823 | 0.349 | 0.382 | 0.788 |
| K-Nearest Neighbors | 0.832 | 0.381 | 0.416 | 0.766 |

### Initial Model Selection

**Gradient Boosting** was selected for further optimization because it achieved the highest F2-score among the compared models.

Although K-Nearest Neighbors achieved higher accuracy, its substantially lower recall made it less suitable for the project's retention-oriented objective.

This demonstrates why model selection should reflect business priorities rather than accuracy alone.

## 6. Gradient Boosting Hyperparameter Optimization

The selected Gradient Boosting model was optimized using `RandomizedSearchCV` and stratified five-fold cross-validation.

The tuning procedure evaluated 40 parameter combinations across five folds, resulting in 200 model fits.

### Selected Hyperparameters

| Hyperparameter | Value |
|---|---:|
| `n_estimators` | 400 |
| `learning_rate` | 0.03 |
| `max_depth` | 3 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 2 |
| `subsample` | 1.0 |

**Best Cross-Validated F2-Score:** 0.6754

### Final Model Performance

| Metric | Baseline GBM | Tuned GBM |
|---|---:|---:|
| Accuracy | 0.805 | 0.803 |
| Precision | 0.516 | 0.511 |
| Recall | 0.722 | 0.727 |
| F1-Score | 0.602 | 0.600 |
| F2-Score | 0.668 | 0.670 |
| ROC-AUC | 0.857 | 0.858 |

The tuned model achieved:

- **Accuracy:** 80.27%
- **Precision:** 51.09%
- **Recall:** 72.67%
- **F2-Score:** 0.6701
- **ROC-AUC:** 0.8579

Hyperparameter tuning produced a modest improvement in recall, F2-score, and ROC-AUC, while slightly reducing accuracy and precision.

These results illustrate the trade-offs involved in optimizing classification models for business-specific objectives.

## 7. Model Explainability Using SHAP

SHAP was applied to the tuned Gradient Boosting model to interpret the factors contributing to customer churn predictions.

The analysis used a SHAP beeswarm plot and feature importance visualization.

### Important Churn Predictors

**1. Age**

Age emerged as the strongest predictor.

Higher transformed age values generally contributed toward higher predicted churn risk, while younger customers tended to receive contributions toward remaining with the bank.

**2. Number of Products**

Product holdings exhibited a nonlinear relationship with churn.

Customers holding two products generally had lower churn rates, while customers holding three or four products showed substantially higher rates.

**3. Customer Activity**

Inactive customers were more likely to receive positive contributions toward churn predictions.

Active membership generally reduced predicted churn risk.

**4. Account Balance**

Higher account balances tended to contribute toward higher predicted churn probabilities.

This finding suggests that customers with substantial balances may represent an important retention segment.

**5. Geographic Location**

Customers in Germany generally received higher predicted churn contributions.

However, geography should be interpreted alongside account balances and other customer characteristics rather than treated as a direct cause of churn.

### Important Interpretation

SHAP values explain how features contribute to model predictions.

They do not establish that changing a feature would directly cause a customer's churn probability to increase or decrease.

## 8. Business Recommendations

The project translates predictive insights into customer retention interventions.

### Strategy 1: Product-Fit Intervention

**Key Predictor:** `NumOfProducts`

Customers with high predicted churn risk and potentially unsuitable product combinations may be offered:

- Product portfolio reviews
- Account and fee assessments
- Simplification of unnecessary products
- More suitable banking product recommendations

### Strategy 2: Customer Re-Engagement

**Key Predictor:** `IsActiveMember`

Inactive customers with elevated predicted churn risk may receive:

- Personalized communications
- Customer service check-ins
- Relevant loyalty or engagement offers
- Targeted re-engagement campaigns

### Strategy 3: High-Value Customer Retention

**Key Predictor:** `Balance`

Customers with high account balances and elevated predicted churn risk may be prioritized for:

- Relationship manager outreach
- Account reviews
- Personalized financial service offers
- Service recovery initiatives

### Strategy 4: Context-Specific Retention

**Relevant Variables:** Age, Geography, Credit Score, and Tenure

These characteristics may provide additional context for understanding customer needs and tailoring appropriate retention services.

The project recommends using these factors responsibly rather than automatically excluding or disadvantaging particular customer groups.

## 9. Business Value

| Analytical Output | Potential Business Application |
|---|---|
| Churn probability | Identify customers for proactive retention |
| Recall-weighted model selection | Reduce missed churners |
| SHAP explanations | Understand predicted risk factors |
| Product analysis | Identify potential product-fit issues |
| Customer engagement analysis | Support targeted re-engagement |
| Account balance analysis | Prioritize valuable customer relationships |

The analysis demonstrates how supervised machine learning can support data-informed retention planning.

Actual business effectiveness would require testing retention interventions and measuring their incremental impact.

## 10. Tools and Technologies

**Programming Language**
- Python

**Data Processing**
- Pandas
- NumPy

**Visualization**
- Matplotlib
- Seaborn

**Machine Learning**
- Scikit-learn
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine
- Gaussian Naive Bayes
- K-Nearest Neighbors

**Model Optimization**
- GridSearchCV
- RandomizedSearchCV
- Stratified K-Fold Cross-Validation

**Model Explainability**
- SHAP

**Development Environment**
- Google Colab
- Jupyter Notebook

## 11. How to Reproduce the Analysis

1. Download or clone this repository.
2. Open `Grp_1_Supervised.ipynb` using Google Colab or Jupyter Notebook.
3. Install the required Python libraries.
4. Run the notebook cells sequentially.
5. The dataset is retrieved using `kagglehub`.
6. Review the exploratory analysis and model comparisons.
7. Execute the hyperparameter tuning procedure.
8. Generate SHAP visualizations and review the retention recommendations.

### Dataset Loading

```python
import kagglehub
import pandas as pd
import os

path = kagglehub.dataset_download(
    "saurabhbadole/bank-customer-churn-prediction-dataset"
)

churn_df = pd.read_csv(
    os.path.join(path, "Churn_Modelling.csv")
)
```

Some model training procedures may require additional computational time, particularly SVM optimization and Gradient Boosting hyperparameter tuning.

## 12. Project Limitations

- The dataset may not represent the characteristics of current banking customers.
- The churn outcome is imbalanced, with approximately 20% positive observations.
- Model performance was evaluated using the available dataset without external validation.
- Predictive associations do not establish causal relationships.
- The high churn rates among customers holding three or four products are based on relatively small groups.
- The project does not evaluate the actual financial cost or effectiveness of retention interventions.
- Model predictions should be monitored for fairness and potential differences in performance across customer groups.
- Additional validation would be required before deploying the model in a production banking environment.

## 13. Future Improvements

Potential extensions include:

1. Optimizing the classification threshold based on retention costs and expected customer value.
2. Evaluating probability calibration.
3. Performing repeated or nested cross-validation.
4. Assessing fairness and subgroup performance.
5. Incorporating customer transaction history and behavioral trends.
6. Estimating customer lifetime value.
7. Evaluating retention interventions through controlled experiments.
8. Developing an interactive churn prediction dashboard.
9. Building an automated data pipeline for periodic model updates.

## 14. Conclusion

This project demonstrates how supervised machine learning can help identify customers at risk of leaving a financial institution.

Six classification algorithms were evaluated, with **Gradient Boosting selected based on its recall-weighted F2-score**.

The tuned model achieved an F2-score of 0.6701, recall of 72.67%, and ROC-AUC of 0.8579.

SHAP analysis identified age, number of products, account balance, customer activity, and geography as important factors influencing model predictions.

The findings were translated into proposed product-fit, re-engagement, and high-value customer retention strategies.

The project highlights the importance of aligning machine learning model selection with business objectives and combining predictive accuracy with model interpretability.

## 15. Project Contributors

**Project Type:** Academic Group Project (Collaborative Work)

This project was **co-developed by four group members** as part of an academic supervised machine learning exercise. The team collaboratively worked on developing and evaluating machine learning models for bank customer churn prediction, interpreting model results, and translating analytical findings into potential customer retention strategies.

### Team Members

| Contributor | Project Role |
|---|---|
| **Drea** | Co-developer and Project Contributor |
| **Gids** | Co-developer and Project Contributor |
| **Dani** | Co-developer and Project Contributor |
| **Mico** | Co-developer and Project Contributor |

### Collaborative Project Scope

The project encompasses the following analytical activities:

- Data preparation, preprocessing, and exploratory data analysis
- Development and comparison of six supervised machine learning algorithms
- Hyperparameter tuning and model performance evaluation
- Gradient Boosting model optimization
- SHAP-based model explainability
- Interpretation of customer churn predictors
- Development of proposed customer retention strategies
- Documentation and presentation of analytical findings

**Contribution Note:** This repository showcases work co-developed by the team. The analytical methods, results, and conclusions are presented as collaborative project outputs. Individual responsibilities have not been separately attributed because the division of work for this project has not been specified.

## 16. References and Data Source

**Bank Customer Churn Prediction Dataset**

https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset

**Scikit-learn Documentation**

https://scikit-learn.org/stable/

**SHAP Documentation**

https://shap.readthedocs.io/

---

*This project was developed for academic and educational purposes. The results demonstrate predictive modeling techniques and proposed retention strategies, rather than validated production banking decisions.*
