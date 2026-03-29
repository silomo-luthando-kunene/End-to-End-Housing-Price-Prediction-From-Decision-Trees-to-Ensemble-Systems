# 🏠 House Price Prediction using Ensemble Machine Learning

### 📌 **Executive Summary** <br/>
This project focuses on predicting residential property prices using structured housing data. The objective was to build a **robust, reusable machine learning pipeline** and evaluate how different models, particularly **ensemble methods**, improve predictive performance. <br/>
<br/> **The project emphasises:**
- End to end ML workflow  
- Preprocessing pipelines  
- Comparative analysis of ensemble model performance 
- Model evaluation and error diagnostics  

### 🎯 **Problem Statement** <br/>
In the real estate market, static pricing models often fail to account for the complex, non-linear relationships between a property’s physical attributes and its market value. This is a **stochastic optimization problem**: how can we build a predictive system that minimizes "Cost of Error" (MAE) across diverse price brackets from budget housing to luxury estates, without our models over-fitting to noise? <br/>
This project aims to train and compare accurate models to predict residential house sale prices (`SalePrice`) in Ames, Iowa, using a dataset containing numerical and categorical features describing residential properties. <br/>

## 🧠 Methodology <br/>
### 1. Data Preprocessing
- Handled missing values using appropriate imputation strategies (SimpleImputer() was not used - imputation was hard coded for the relevant columns using .fillna())
- Split features into:
  - Numerical variables
  - Categorical variables
- Applied [sklearn.preprocessing]:
  - `StandardScaler` for numerical features 
  - `OneHotEncoder` for categorical features to turn categorical data into numerical data <br/>

### 2. Pipeline Engineering (Core Focus 🚀)
To ensure scalability and reproducibility: <br/>
- Built preprocessing workflows using `ColumnTransformer` using the Scikit Learn Library [Compose module]
- Integrated preprocessing and modeling using `Pipeline` using the Scikit Learn Library [Pipeline module]
- Enabled efficient experimentation across multiple models without duplicating code <br/>

### 3. Models Implemented

#### 🌳 Baseline Model
- Decision Tree Regressor

#### 🌲 Bagging
- Random Forest Regressor

#### 🚀 Boosting
- Gradient Boosting Regressor

#### 🤝 Ensemble Methods
- Voting Regressor (combining Random Forest Regressor and Gradient Boost Regressor models)
- Stacking Regressor (Linear Regression meta-model approach incorporating Random Forest Regressor and Gradient Boost Regressor models) <br/>

## 📊 Model Performance and 📉 Evaluation

| Model | MAE | R² | MAPE (Overall) | 
|------|------|----|---|
| Decision Tree | ~28290 | ~0.7504 | ~16.16% | 
| Random Forest | ~17640 | ~0.8909 | ~10.77% |
| Gradient Boosting | ~18660 | ~0.8881 | ~11.27% |
| Voting Regressor | ~17455 | ~0.8951 | ~10.61% |
| Stacking Regressor | ~17679 | ~0.8985 | ~10.54% |

> Decision Tree model's average prediction error is off by $28 290 in comparison to the Voting Regressor's which is $17 455. The Voting Regressor improves accuracy, lowering the average error by roughly 38% compared to the Decision Tree baseline. 
> Ensemble models significantly outperformed the baseline model in both accuracy and generalisation. <br/>

The following metrics were calculated:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Root Mean Squared Log Error (RMSLE)
- R² Score (R2) <br/>

Additional evaluation included:
- Residual analysis
- Performance breakdown (MAPE%) by price segments <br/>

## 📊 Visual Analysis

### 🔹 Feature Importance Comparison
> Shows how different models prioritise features differently

<p align="left">
  <img src="https://github.com/silomo-luthando-kunene/End-to-End-Housing-Price-Prediction-From-Decision-Trees-to-Ensemble-Systems/blob/main/Visuals/Agg%20Models%20Feature%20Importance%20Matrix.png?raw=true" 
       alt="Feature Importance" 
       width="65%" />
</p>

---

### 🔹 Actual vs Predicted Prices
> Evaluates how closely predictions align with real values
> There is clear improvement from the base model to the best model used as seen by the clusters tucking closer to the line of best of fit showing that the model's predictions are not far off from the actual values and that there is low bias.

<p align="left">
  <img src="https://github.com/silomo-luthando-kunene/End-to-End-Housing-Price-Prediction-From-Decision-Trees-to-Ensemble-Systems/blob/main/Visuals/Decision%20Tree%20Scatter%20Plot.png?raw=true" 
       alt="Stacking Regressor Scatter Plot" 
       width="49%" />
  <img src="https://github.com/silomo-luthando-kunene/End-to-End-Housing-Price-Prediction-From-Decision-Trees-to-Ensemble-Systems/blob/main/Visuals/Stacking%20Regressor%20Scatter%20Plot.png?raw=true" 
       alt="Decision Tree Scatter Plot" 
       width="49%" />
</p>

---

### 🔹 Error by Price Bracket (Business Insight)
> Highlights model performance across different housing segments

<p align="left">
  <img src="https://github.com/silomo-luthando-kunene/End-to-End-Housing-Price-Prediction-From-Decision-Trees-to-Ensemble-Systems/blob/main/Visuals/Decision%20Tree%20MAPE%20Bins.png?raw=true" 
       alt="Stacking Regressor MAPE Bins" 
       width="65%" />
  <img src="https://github.com/silomo-luthando-kunene/End-to-End-Housing-Price-Prediction-From-Decision-Trees-to-Ensemble-Systems/blob/main/Visuals/Stacking%20Regressor%20MAPE%20Bins.png?raw=true" 
       alt="Decision Tree MAPE Bins" 
       width="65%" />
</p>
<br/>
🔍 **Key Insights**

- Ensemble learning significantly improved performance over a single Decision Tree
- Different models captured **different feature relationships**
- Voting and Stacking models provided **more stable predictions**
- Model performance varied across price ranges:
  - Strong performance in mid-range properties  
  - Higher error in high-value properties (likely due to limited data representation) <br/>

## 🧠 Key Learnings

- Built reusable ML workflows using `Pipeline` and `ColumnTransformer`
- Developed understanding of:
  - Bagging vs Boosting [Homogenous Ensemble Methods]
  - Voting vs Stacking ensembles [Heterogenous Ensemble Methods]
- Learned to:
  - Evaluate models using multiple metrics
  - Diagnose model weaknesses through error analysis
  - Interpret feature importance across models
- Strengthened ability to structure **end-to-end ML projects** <br/>

## 🚀 Reflections
*   **Process Efficiency:** Moving from manual data cleaning to **Scikit-Learn Pipelines** reduced technical debt and made the model iteration cycle 3x faster.
*   **Fail-Safe Ingestion:** Encountering `NaN` values in the unseen `test.csv` provided a critical learning moment. Future iterations will involve integrating the **SimpleImputer** directly into the `ColumnTransformer` to handle "noisy" real-world data automatically.

**NB:** <br/>
This was done in tandem with learning material covered in the past 7 days in relation to a Machine Learning course I am currently enrolled in with ALX Africa (Explore AI Academy) - This project is my application and active learning of Machine Learning.
