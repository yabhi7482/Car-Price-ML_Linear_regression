# 🚗 Car Price Prediction using Machine Learning

A Machine Learning project that predicts the **price of used cars** based on different features such as brand, model, mileage, engine volume, car body, engine type, registration status, and car age.

The project covers the complete Machine Learning workflow from **data collection and EDA to preprocessing, feature engineering, model training, evaluation, and model saving**.

---

## 📌 Project Overview

Used car prices depend on several factors such as:

* 🚘 Car Brand
* 🏷️ Car Model
* ⛽ Mileage
* 🔧 Engine Volume
* 📅 Car Age
* 🚗 Body Type
* ⚙️ Engine Type
* 📋 Registration Status

The goal of this project is to build a Machine Learning model that can learn these relationships and predict the price of a used car.

---

## 🎯 Objective

The main objective of this project is to:

* Understand the used-car dataset
* Perform Exploratory Data Analysis (EDA)
* Handle missing values
* Detect and correct unrealistic values
* Transform skewed numerical features
* Encode categorical features
* Train a Machine Learning model
* Evaluate model performance
* Save the trained model for future predictions

---

## 📂 Dataset

**Dataset:** Used Car Sales Data

The dataset contains information about used cars and their selling prices.

### Main Features

| Feature        | Description                 |
| -------------- | --------------------------- |
| `Brand`        | Car manufacturer/brand      |
| `Model`        | Car model                   |
| `Price`        | Car price — Target Variable |
| `Body`         | Body type of the car        |
| `Mileage`      | Mileage of the car          |
| `EngineV`      | Engine volume               |
| `Engine Type`  | Type of engine              |
| `Registration` | Registration status         |
| `Year`         | Manufacturing year          |

---

# 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Exploration
   ↓
Missing Value Handling
   ↓
Data Cleaning
   ↓
Outlier / Data Entry Error Handling
   ↓
Feature Transformation
   ↓
Categorical Encoding
   ↓
Train-Test Split
   ↓
Linear Regression
   ↓
Model Evaluation
   ↓
Save Model
```

---

# 🛠️ Technologies & Libraries

The project is developed using Python.

### Programming Language

* Python 🐍

### Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* KaggleHub

---

# 🔎 Exploratory Data Analysis

Before training the Machine Learning model, the dataset was explored using:

### Dataset Information

```python
df.shape
df.info()
df.describe()
```

### Missing Value Analysis

```python
df.isnull().sum()
```

### Distribution Analysis

Histograms were used to understand the distributions of:

* Price
* Mileage
* Engine Volume
* Car Age

### Correlation Analysis

A correlation heatmap was created to understand the relationship between numerical features.

---

# 🧹 Data Cleaning

Several data-cleaning techniques were applied before training the model.

## 1. Missing Values

Missing values in numerical columns were handled using appropriate statistical values such as the mean and median.

Example:

```python
df['EngineV'] = df['EngineV'].fillna(df['EngineV'].mean())
```

---

## 2. Engine Volume Correction

Some `EngineV` values were unrealistic because of data-entry errors.

For example:

```text
4.5 → correct
45  → incorrect
65  → incorrect
```

Values greater than `6.5` were checked and corrected by dividing them by `10`.

```python
df.loc[df['EngineV'] > 6.5, 'EngineV'] = (
    df.loc[df['EngineV'] > 6.5, 'EngineV'] / 10
)
```

This helped reduce unrealistic engine-volume values.

---

# 📊 Feature Transformation

## Mileage Transformation

Mileage was right-skewed, so a logarithmic transformation was applied.

```python
df['Mileage'] = np.log(df['Mileage'])
```

This helped make the distribution more balanced.

---

## Car Age Transformation

Instead of directly using the manufacturing year, car age was calculated:

```python
df['Year'] = 2026 - df['Year']
```

Then `log1p()` transformation was applied:

```python
df['Year'] = np.log1p(df['Year'])
```

This converts the original manufacturing year into a more meaningful feature representing **car age**.

---

## Price Transformation

The target variable `Price` was highly right-skewed.

Therefore, logarithmic transformation was applied:

```python
df['Price'] = np.log(df['Price'])
```

This helps reduce the effect of extremely high prices and makes the target distribution more suitable for Linear Regression.

---

# 🔤 Feature Encoding

Machine Learning models require numerical input, so categorical variables were converted into numerical form.

## Model Frequency Encoding

The `Model` column contains many unique car models.

Instead of creating hundreds of dummy variables, frequency encoding was used.

```python
model_frequency = (
    df_encode['Model']
    .value_counts(normalize=True)
    .to_dict()
)

df_encode['Model'] = (
    df_encode['Model']
    .map(model_frequency)
)
```

This converts each car model into its occurrence frequency.

---

## One-Hot Encoding

Other categorical columns were converted using One-Hot Encoding.

```python
df_encode = pd.get_dummies(
    df_encode,
    columns=[col],
    drop_first=True,
    dtype=int
)
```

This was applied to categorical features such as:

* Brand
* Body
* Engine Type
* Registration

---

# 🧮 Handling Infinite and Missing Values

Infinite values were replaced with `NaN`:

```python
df_encode = df_encode.replace(
    [np.inf, -np.inf],
    np.nan
)
```

Missing values were then filled using the median:

```python
df_encode = df_encode.fillna(
    df_encode.median()
)
```

---

# ✂️ Train-Test Split

The dataset was divided into training and testing sets.

```python
X = df_encode.drop(columns=['Price'])
y = df_encode['Price']
```

Then:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
```

### Split

* **70% → Training Data**
* **30% → Testing Data**

---

# 🤖 Machine Learning Model

## Linear Regression

Linear Regression was selected as the baseline prediction model.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
```

Predictions were generated using:

```python
y_predict = model.predict(X_test)
```

---

# 📈 Model Evaluation

The model was evaluated using three important metrics.

### Mean Absolute Error — MAE

Measures the average absolute difference between actual and predicted values.

```python
mae = mean_absolute_error(
    y_test,
    y_predict
)
```

### Root Mean Squared Error — RMSE

Penalizes larger prediction errors more strongly.

```python
rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_predict
    )
)
```

### R² Score

Measures how well the model explains the variation in the target variable.

```python
r2 = r2_score(
    y_test,
    y_predict
)
```

### Adjusted R²

Adjusted R² was also calculated while considering the number of features used by the model.

```python
n = X_test.shape[0]
p = X_test.shape[1]

adjusted_r2 = 1 - (
    (1 - r2) * (n - 1) /
    (n - p - 1)
)
```

---

# 💾 Model Saving

The trained model was saved using Joblib.

```python
import joblib

joblib.dump(
    model,
    'Car_price_model.pkl'
)
```

The saved model can later be loaded and used for prediction without retraining the model.

```python
model = joblib.load(
    'Car_price_model.pkl'
)
```

---

# 📁 Project Structure

```text
Car_Price/
│
├── EDA_for_Car_prediction.ipynb
├── Car_price_model.pkl
├── main.py
├── README.md
│
└── dataset/
    └── car_sales.csv
```

> Your actual file names/folders can be changed according to your GitHub repository.

---

# 🚀 Future Improvements

The project can be improved further by:

* Trying Random Forest Regression
* Trying Gradient Boosting
* Trying XGBoost
* Hyperparameter tuning
* Cross-validation
* Better outlier detection
* Using a complete preprocessing pipeline
* Building an interactive Streamlit application
* Deploying the prediction application online

---

# 💡 Key Learnings

Through this project, I learned:

* How to perform EDA on real-world data
* How to identify missing values
* How to handle data-entry errors
* How to detect skewed distributions
* How to apply logarithmic transformations
* Frequency Encoding
* One-Hot Encoding
* Train-Test Split
* Linear Regression
* MAE, RMSE and R² evaluation
* Model saving using Joblib
* Complete Machine Learning workflow

---

# 📌 Conclusion

This project demonstrates an end-to-end **Used Car Price Prediction Machine Learning workflow**.

Starting from raw data, the project performs:

```text
Data Collection
      ↓
EDA
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Feature Encoding
      ↓
Data Transformation
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Saving
```

The project helped me understand how real-world data is cleaned, transformed, encoded, and prepared before applying Machine Learning algorithms.

---

## 👨‍💻 Author

**Abhishek Yadav**

B.Tech CSE | Data Science & Machine Learning Enthusiast

---

## ⭐ If you found this project useful

Feel free to ⭐ **Star** this repository and explore the project.
