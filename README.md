# 📜 박예지 포트폴리오

> 박예지(Yeji Park) - Yeji Park's Portfolio

<br />




<h1 align="center">Hi 👋, I'm Yeji Park</h1>
<h3 align="center">A Data Analytics student from Pennsylvania State University</h3>
<h3 align="center">Languages and Tools:</h3>
<p align="center"> <a href="https://www.mysql.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/mysql/mysql-original-wordmark.svg" alt="mysql" width="40" height="40"/> </a> <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> </p>


<h1 align="left">Projects</h1>

<h2 align="left"> Project 1: CO2 Emissions Trends Forecast on Carbon Tax</h2>
- 🔭 I’m currently working on **CO2 Emissions Trends Forecast on 3 environmental policies.**


<h2>Data Pre-Processing:</h2>

<strong>Data Cleaning:</strong>
- Conducted comprehensive removal of missing values (NAs) to ensure data integrity and accuracy
- Employed rigorous data validation techniques to identify and eliminate unreliable or inconsistent data points

<strong>Feature Selection:</strong>
- Implemented strategic removal of unnecessary columns to streamline the dataset and enhance computational efficiency
- Leveraged domain knowledge and statistical analysis to identify and retain only the most relevant features for analysis

<strong>Column Standardization:</strong>
- Executed systematic column renaming to establish a standardized naming convention
- Enhanced dataset clarity and interpretability by assigning descriptive and intuitive names to each column

<strong>Data Integration:</strong>
- Performed seamless merging of disparate datasets to create a unified and comprehensive analytical framework
- Incorporated advanced data integration techniques to combine information from multiple sources and enhance the depth and breadth of analysis

<h2>Data Modeling:</h2>

<strong>a. Panel OLS</strong>
- Utilized Panel OLS regression to analyze the effect of carbon tax policy
- Examined the relationship between carbon tax implementation and CO2 emissions across multiple countries
- Incorporated time-series and cross-sectional data to account for both within-country and between-country variations

<strong>b. Exponential Smoothing</strong>
- Applied Exponential Smoothing technique for forecasting CO2 emissions trends
- Captured the underlying patterns and trends in the data while minimizing the impact of random fluctuations
- Utilized this method to generate short-term forecasts of CO2 emissions based on historical data

<strong>c. XGBoost</strong>
- Implemented XGBoost algorithm for predicting future CO2 emissions
- Leveraged the ensemble learning technique to build a powerful predictive model
- Tuned hyperparameters and optimized the model's performance to achieve accurate forecasts

<h2 align="left"> Project 2: Plagiarism Detection</h2>

Project 1: 유방암 분류 

**Background** 

이 프로젝트는 SVM (Support Vector Machine) 및 SVC (Support Vector Classification) 알고리즘을 사용하여 유방암 분류를 예측하는 데 활용되었습니다. 데이터는 **`sklearn`** 라이브러리의 **`load_breast_cancer`** 함수를 통해 수집되었습니다. 이 프로젝트는 모델 향상뿐만 아니라 SVM 및 SVC를 사용하여 예측 모델링에 초점을 맞추었습니다.

**Summary**

**(1). Data Collection**

- 수집 대상: 유방암 관련 데이터
- 수집 출처: **`sklearn.datasets`** 라이브러리의 **`load_breast_cancer`** 함수 활용

**(2). Model & Algorithms**

- 머신러닝 모델: SVM 및 SVC 알고리즘을 사용하여 유방암 분류에 적용
- 주요 특성: **`load_breast_cancer` 함수로 불러온 데이터의 특성 활용**
- 모델 향상: SVM Parameter Optimization, SVC의 gamma 매개변수 활용
- 데이터 정규화: 데이터 스케일링 수행

**(3). Model Evaluation**
- **정확성 (Accuracy):** 96%
- **정밀도 (Precision):**
    - 클래스 0: 100%
    - 클래스 1: 94%
- **재현율 (Recall):**
    - 클래스 0: 92%
    - 클래스 1: 100%
- **F1 점수 (F1-score):**
    - 클래스 0: 96%
    - 클래스 1: 97%

**분석 및 해석:**

1. **정확성 96%:** 모델이 전체 데이터에서 정확하게 분류한 비율이 96%입니다. 
2. **클래스 별 성능:**
    - **클래스 0:** 정확도 100%와 높은 정밀도, 재현율, F1 점수는 이 클래스에서 모델이 매우 효과적으로 예측을 수행했음을 나타냅니다.
    - **클래스 1:** 정확도 94%와 높은 정밀도, 재현율, F1 점수는 모델이 양성 클래스도 효과적으로 예측했음을 나타냅니다.
