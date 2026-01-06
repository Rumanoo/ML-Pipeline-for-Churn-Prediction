import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def main():
    # Load dataset
    df = pd.read_csv("data/Telco-Customer-Churn.csv")

    # Target encoding
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df.drop(columns=['customerID'], inplace=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Preprocessing
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Logistic Regression pipeline
    log_reg_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    log_reg_params = {
        'classifier__C': [0.1, 1, 10]
    }

    log_reg_grid = GridSearchCV(
        log_reg_pipeline,
        log_reg_params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    log_reg_grid.fit(X_train, y_train)

    # Random Forest pipeline
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    rf_params = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20]
    }

    rf_grid = GridSearchCV(
        rf_pipeline,
        rf_params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    rf_grid.fit(X_train, y_train)

    # Evaluation
    print("Logistic Regression Performance:")
    print(classification_report(y_test, log_reg_grid.predict(X_test)))

    print("\nRandom Forest Performance:")
    print(classification_report(y_test, rf_grid.predict(X_test)))

    # Save best model
    best_model = rf_grid.best_estimator_
    joblib.dump(best_model, "models/telco_churn_pipeline.pkl")

    print("\nPipeline saved successfully!")


if __name__ == "__main__":
    main()
