
import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier


# ---------------------------------------------------------
# 1. Load train and test data
# ---------------------------------------------------------

DATA_DIR = "tourism_project/model_building"

X_train = pd.read_csv(
    os.path.join(DATA_DIR, "X_train.csv")
)

X_test = pd.read_csv(
    os.path.join(DATA_DIR, "X_test.csv")
)

y_train = pd.read_csv(
    os.path.join(DATA_DIR, "y_train.csv")
).squeeze()

y_test = pd.read_csv(
    os.path.join(DATA_DIR, "y_test.csv")
).squeeze()

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ---------------------------------------------------------
# 2. Identify categorical and numerical columns
# ---------------------------------------------------------

categorical_columns = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# ---------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ]
)


# ---------------------------------------------------------
# 4. Define XGBoost model
# ---------------------------------------------------------

model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)


# ---------------------------------------------------------
# 5. Create preprocessing + model pipeline
# ---------------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ---------------------------------------------------------
# 6. Define hyperparameter grid
# ---------------------------------------------------------

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1.0]
}


# ---------------------------------------------------------
# 7. Hyperparameter tuning
# ---------------------------------------------------------

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=1
)


# ---------------------------------------------------------
# 8. MLflow experiment tracking
# ---------------------------------------------------------

mlflow.set_experiment(
    "Visit_with_Us_Wellness_Tourism"
)

with mlflow.start_run():

    print("\nStarting hyperparameter tuning...")

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    best_params = grid_search.best_params_

    print("\nBest Parameters:")
    print(best_params)


    # -----------------------------------------------------
    # 9. Evaluate the best model
    # -----------------------------------------------------

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print("\nModel Performance:")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)


    # -----------------------------------------------------
    # 10. Log parameters and metrics to MLflow
    # -----------------------------------------------------

    mlflow.log_params(best_params)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        best_model,
        name="best_xgboost_model"
    )


# ---------------------------------------------------------
# 11. Save the best model
# ---------------------------------------------------------

DEPLOYMENT_DIR = "tourism_project/deployment"

os.makedirs(
    DEPLOYMENT_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    DEPLOYMENT_DIR,
    "best_model.pkl"
)

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nBest model saved successfully at:")
print(MODEL_PATH)
