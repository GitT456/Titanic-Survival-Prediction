# 🚢 Titanic Survival Prediction

A complete machine learning project for predicting passenger survival in the **Titanic - Machine Learning from Disaster** competition.  
The goal of this project is to use passenger information such as ticket class, gender, age, fare, family relations, and other available features to predict whether a passenger survived or not.

This notebook includes data loading, exploratory data analysis, data cleaning, preprocessing, model comparison, final model training, result visualization, and creation of the final `submission.csv` file for Kaggle.

---

## 📌 Project Overview

The Titanic dataset is one of the most popular beginner-friendly machine learning datasets.  
In this project, the main task is a **binary classification problem**:

| Target | Meaning |
|---|---|
| `0` | Passenger did not survive |
| `1` | Passenger survived |

The dataset is divided into two main files:

| File | Description |
|---|---|
| `train.csv` | Training data. It includes passenger information and the real survival result. |
| `test.csv` | Test data. It includes passenger information, but the survival result is hidden. |
| `gender_submission.csv` | A sample submission file showing the required Kaggle submission format. |

The model is trained on `train.csv` and then used to predict survival values for passengers in `test.csv`.

---

## 🎯 Project Goal

The main objectives of this project are:

- Understand the Titanic dataset.
- Clean and preprocess the data.
- Prepare train and test features correctly.
- Compare multiple machine learning algorithms.
- Select the best-performing model based on Cross Validation.
- Train the final model on the full training dataset.
- Generate a valid `submission.csv` file for Kaggle.

---

## 📊 Dataset Description

The dataset contains passenger-related information from the Titanic disaster.

### Data Dictionary

| Variable | Definition | Values / Notes |
|---|---|---|
| `PassengerId` | Unique passenger identifier | Numeric ID |
| `Survived` | Survival status | `0 = No`, `1 = Yes` |
| `Pclass` | Ticket class | `1 = 1st`, `2 = 2nd`, `3 = 3rd` |
| `Name` | Passenger name | Text |
| `Sex` | Passenger gender | Male / Female |
| `Age` | Age in years | Numeric |
| `SibSp` | Number of siblings/spouses aboard | Numeric |
| `Parch` | Number of parents/children aboard | Numeric |
| `Ticket` | Ticket number | Text |
| `Fare` | Passenger fare | Numeric |
| `Cabin` | Cabin number | Text / Missing values |
| `Embarked` | Port of embarkation | `C = Cherbourg`, `Q = Queenstown`, `S = Southampton` |

### Variable Notes

| Variable | Explanation |
|---|---|
| `Pclass` | A proxy for socio-economic status. 1st class usually represents upper class, 2nd middle class, and 3rd lower class. |
| `Age` | Some values are missing and need to be handled during preprocessing. |
| `SibSp` | Represents siblings and spouses traveling with the passenger. |
| `Parch` | Represents parents and children traveling with the passenger. |
| `Cabin` | Contains many missing values, so it should be handled carefully. |
| `Embarked` | Represents the port where the passenger boarded the ship. |

---

## 🧹 Data Preprocessing

The preprocessing stage is one of the most important parts of this project.  
The goal was to prepare both train and test data in a consistent and clean format.

Main preprocessing steps:

1. **Handling missing values**
   - Numeric missing values were filled using appropriate statistical values such as median.
   - Categorical missing values were filled using the most frequent value or encoded after cleaning.

2. **Encoding categorical features**
   - Text-based columns such as gender or embarkation port were converted into numeric values so machine learning models could use them.

3. **Feature and target separation**
   - `Survived` was selected as the target variable.
   - The remaining useful columns were used as model features.

4. **Train/Test consistency**
   - The train and test datasets were processed in the same structure.
   - Features used for training were aligned with features used for prediction.

5. **Removing non-useful identifiers**
   - `PassengerId` was kept for the final submission file.
   - It was not used as a meaningful learning feature because it is only an identifier.

---

## 🔎 Exploratory Data Analysis

Exploratory Data Analysis was used to better understand the dataset before training the model.

The notebook includes analysis such as:

- Dataset shape and overview.
- Missing value inspection.
- Distribution of important variables.
- Relationship between passenger features and survival.
- Survival patterns based on gender, passenger class, age, fare, and family-related variables.

EDA helped identify which features may have stronger relationships with survival.

---

## 🤖 Models Tested

Several machine learning algorithms were tested using the same processed dataset.

The tested models included:

| Model | Type |
|---|---|
| Logistic Regression | Linear classification model |
| KNN | Distance-based classification model |
| SVM | Margin-based classification model |
| Random Forest | Ensemble tree-based model |
| Extra Trees | Ensemble randomized tree model |
| Gradient Boosting | Boosting-based ensemble model |
| AdaBoost | Boosting-based ensemble model |
| XGBoost | Advanced gradient boosting model |
| CatBoost | Gradient boosting model, if available |

The models were compared using **5-Fold Cross Validation**.

---

## 🏆 Final Model: Gradient Boosting Classifier

After comparing multiple algorithms, **Gradient Boosting Classifier** achieved the best Cross Validation performance.

### Why Gradient Boosting?

Gradient Boosting is an ensemble learning method that builds models sequentially.  
Each new model tries to correct the mistakes made by the previous models. This makes it powerful for structured tabular datasets like Titanic.

The final selected model was:

```python
GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
```

### Final Model Result

| Metric | Result |
|---|---:|
| Best Model | Gradient Boosting |
| Best Cross Validation Accuracy | 83.39% |
| Problem Type | Binary Classification |
| Target Column | `Survived` |
| Final Output File | `submission.csv` |

The model was selected because it achieved the highest Cross Validation accuracy among the tested algorithms.

---

## 📈 Model Evaluation

Since the Kaggle `test.csv` file does not contain the real `Survived` values, direct accuracy calculation on the test set is not possible.

For this reason, model evaluation was performed using Cross Validation on the training dataset.

Evaluation included:

- Cross Validation Accuracy
- Model comparison table
- Feature importance plot
- Prediction distribution
- Confusion Matrix using cross-validated predictions

This approach provides a reliable internal estimate of model performance before creating the final Kaggle submission.

---

## 📌 Result Visualization

The notebook includes several visualizations to explain model behavior and results:

| Plot | Purpose |
|---|---|
| Cross Validation Accuracy Plot | Compare model accuracy across algorithms |
| Feature Importance Plot | Show which features contributed most to prediction |
| Confusion Matrix | Analyze correct and incorrect predictions internally |
| Prediction Distribution | Show the distribution of predicted survival values on test data |

These plots make the notebook more understandable and presentation-ready.

---

## 📤 Creating the Kaggle Submission File

Kaggle requires the final submission file to contain exactly two columns:

| Column | Description |
|---|---|
| `PassengerId` | Passenger ID from `test.csv` |
| `Survived` | Predicted survival value, either `0` or `1` |

The final submission file is created using:

```python
submission = pd.DataFrame({
    "PassengerId": te_df["PassengerId"],
    "Survived": y_pred_gb
})

submission.to_csv("submission.csv", index=False)
```

The output file:

```text
submission.csv
```

can be submitted directly to Kaggle.

Example format:

```text
PassengerId,Survived
892,0
893,1
894,0
...
```

---

## 🗂️ Project Structure

A simple project structure can be:

```text
Titanic-Survival-Prediction/
│
├── input/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
│
├── predict_gradient_boosting_final.ipynb
├── submission.csv
└── README.md
```

---

## ⚙️ Requirements

The main libraries used in this project are:

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
catboost
```

Some models such as CatBoost may require installation if they are not already available:

```python
!pip install catboost
```

For the final Gradient Boosting model, only `scikit-learn` is required.

---

## ▶️ How to Run

### In Google Colab

1. Upload the dataset to Google Drive.
2. Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Set the project directory.
4. Run all notebook cells from top to bottom.
5. The final `submission.csv` file will be created.

### In Kaggle Notebook

1. Open the Titanic competition.
2. Create a new Kaggle Notebook.
3. Add the Titanic competition dataset.
4. Run all cells.
5. Save the notebook version.
6. Submit the generated `submission.csv`.

---

## ✅ Final Conclusion

In this project, the Titanic survival prediction problem was solved as a binary classification task.  
After preprocessing the data and comparing several machine learning algorithms, **Gradient Boosting Classifier** achieved the best Cross Validation result with an accuracy of approximately **83.39%**.

The final model was trained on the complete training dataset and then used to predict survival outcomes for the Kaggle test dataset.  
The predictions were saved in the required Kaggle format as `submission.csv`.

This project demonstrates a complete machine learning workflow:

```text
Data Loading → EDA → Cleaning → Preprocessing → Model Comparison → Final Model → Submission
```

---

## 👤 Author

**Taha Zarei**

Machine Learning Project  
Titanic Survival Prediction using Gradient Boosting
