# 🧹 Complex Data Munging & Statistical Modeling in pandas

## 📌 Project Overview
This project demonstrates **data cleaning, feature engineering, and regression modeling** using a messy employee dataset (salaries, benefits, departments).  
It mimics real-world HR/government datasets where values may be missing, inconsistent, or erroneous.  

The workflow covers:
1. **Data Cleaning** – fixing missing values, inconsistent formats, and invalid salaries  
2. **Feature Engineering** – creating new features such as `Years_of_Service`  
3. **Exploratory Data Analysis (EDA)** – visualizing salary trends, benefits, and service years  
4. **Statistical Modeling** – fitting an **OLS Linear Regression** model with `Salary` as the target  

## 📂 Files
- `data_prep.ipynb` → Data cleaning & feature engineering  
- `modeling.ipynb` → Regression modeling, diagnostics, and interpretation  
- `slides/Final_Presentation.pptx` → Summary slides for presentation  

## ⚙️ Requirements
Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn statsmodels
```

## 🚀 How to Run
1. Clone/download this repository
2. Run all cells in data_prep.ipynb → cleaned dataset will be prepared
3. Run all cells in modeling.ipynb → regression analysis & plots generated
4. View results and insights in slides/Final_Presentation.pptx

## 📊 Modeling Summary
We fit an OLS Linear Regression model:
```
Salary ~ Benefits_Cost + Years_of_Service
```
- Benefits_Cost → Strong positive correlation with salary (statistically significant, p < 0.05)
- Years_of_Service → Weaker impact, not always significant
- Adjusted R² → Shows moderate explanatory power

## 📈 Plots & Diagnostics
Residual Distribution
- Residuals approx. normally distributed
- No strong skewness → regression assumptions hold

Residuals vs Fitted
- Residuals centered around 0
- No clear heteroscedasticity

## ✅ Conclusion
- Benefits_Cost is a key predictor of salary.
- Years_of_Service has limited predictive power.
- Model assumptions hold, making the regression valid.

This project demonstrates the end-to-end process of cleaning messy data, creating features, performing statistical modeling, and communicating insights.