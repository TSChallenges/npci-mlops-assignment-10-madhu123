from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics
import kfp

# Pipeline Component-1: Data Loading & Processing
@component(
    packages_to_install=["pandas==1.5.3", "numpy==1.23.5", "scikit-learn==1.2.2"], # Pinned versions for stability
    base_image="python:3.10-slim",
)
def load_churn_data(drop_missing_vals: bool, churn_dataset: Output[Dataset]):
    """Loads churn data, performs mapping, and optionally drops NAs."""
    import pandas as pd
    import numpy as np # Needed if dropna is used, though unlikely here

    print("Loading and preprocessing data...")
    # Load Customer Churn dataset
    df = pd.read_csv("https://raw.githubusercontent.com/MLOPS-test/test-scripts/refs/heads/main/mlops-ast10/Churn_Modeling.csv")
    print(f"Initial data shape: {df.shape}")

    # Define target variable and features (matching train_model.py)
    X = df[["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "IsActiveMember", "EstimatedSalary"]].copy()
    y = df[["Exited"]] # Keep as DataFrame for easy merge

    print("Mapping categorical features...")
    # Handling category labels present in `Geography` and `Gender` columns
    geography_mapping = {'France': 0, 'Spain': 1, 'Germany': 2}
    gender_mapping = {'Female': 0, 'Male': 1}

    # Map categorical values to numbers using respective dictionaries
    X['Geography'] = X['Geography'].map(geography_mapping)
    X['Gender'] = X['Gender'].map(gender_mapping)
    print("Mapping complete.")

    # Combine features and target for output dataset
    transformed_df = X.copy()
    transformed_df['Exited'] = y['Exited'] # Add the target column

    if drop_missing_vals:
        print("Checking for and dropping missing values...")
        initial_rows = len(transformed_df)
        transformed_df = transformed_df.dropna()
        rows_dropped = initial_rows - len(transformed_df)
        print(f"Dropped {rows_dropped} rows with missing values.")
    else:
        print("Skipping drop missing values step.")

    print(f"Final processed data shape: {transformed_df.shape}")
    # Save the processed data
    transformed_df.to_csv(churn_dataset.path, index=False)
    print(f"Processed data saved to: {churn_dataset.path}")


# Pipeline Component-2: Train-Test Split
@component(
    packages_to_install=["pandas==1.5.3", "numpy==1.23.5", "scikit-learn==1.2.2"], # Pinned versions
    base_image="python:3.10-slim",
)
def train_test_split_churn(
    input_churn_dataset: Input[Dataset],
    X_train: Output[Dataset], # Renamed back
    X_test: Output[Dataset],  # Renamed back
    y_train: Output[Dataset], # Renamed back
    y_test: Output[Dataset],  # Renamed back
    test_size: float,
    random_state: int,
):
    """Splits the data into training and testing sets with stratification."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    print(f"Loading processed data from: {input_churn_dataset.path}")
    df = pd.read_csv(input_churn_dataset.path)
    print(f"Data shape for splitting: {df.shape}")

    if 'Exited' not in df.columns:
        raise ValueError("Target column 'Exited' not found in the input dataset.")

    print(f"Splitting data with test_size={test_size}, random_state={random_state}, stratify=True")
    # Separate features (X) and target (y)
    X_features = df.drop('Exited', axis=1)
    y_target = df['Exited']

    # Perform train-test split using stratification (as in train_model.py)
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_features, y_target,
        test_size=test_size,
        random_state=random_state,
        stratify=y_target  # Crucial for classification balance
    )
    print(f"X_train shape: {X_train_split.shape}, y_train shape: {y_train_split.shape}")
    print(f"X_test shape: {X_test_split.shape}, y_test shape: {y_test_split.shape}")

    # Save the split datasets as CSV files using the output parameter paths
    X_train_split.to_csv(X_train.path, index=False)
    X_test_split.to_csv(X_test.path, index=False)
    # Save y as DataFrame to keep header, consistent with load in next step
    pd.DataFrame(y_train_split).to_csv(y_train.path, index=False)
    pd.DataFrame(y_test_split).to_csv(y_test.path, index=False)

    print(f"X_train saved to: {X_train.path}")
    print(f"X_test saved to: {X_test.path}")
    print(f"y_train saved to: {y_train.path}")
    print(f"y_test saved to: {y_test.path}")


# Pipeline Component-3: Model Training
@component(
    packages_to_install=["pandas==1.5.3", "numpy==1.23.5", "scikit-learn==1.2.2", "joblib==1.2.0"], # Added joblib, Pinned
    base_image="python:3.10-slim",
)
def train_churn_model(
    X_train: Input[Dataset], # Renamed back
    y_train: Input[Dataset], # Renamed back
    model_output: Output[Model],
    n_estimators: int,
    random_state: int,
):
    """Trains a RandomForestClassifier model."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    print(f"Loading training data: X from {X_train.path}, y from {y_train.path}")
    # Use the input parameter names to load data
    X_train_df = pd.read_csv(X_train.path)
    y_train_df = pd.read_csv(y_train.path)
    y_train_series = y_train_df['Exited'] # Extract the Series

    print(f"Training RandomForestClassifier with n_estimators={n_estimators}, random_state={random_state}")
    # Create Random Forest Classifier model
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    # Train the model (using .values.ravel() for y_train as in train_model.py)
    rf_model.fit(X_train_df, y_train_series.values.ravel())
    print("Model training complete.")

    # Save the trained model using joblib
    joblib.dump(rf_model, model_output.path)
    print(f"Trained model saved to: {model_output.path}")


# Pipeline Component-4: Model Evaluation
@component(
    packages_to_install=["pandas==1.5.3", "numpy==1.23.5", "scikit-learn==1.2.2", "joblib==1.2.0"], # Added joblib, Pinned
    base_image="python:3.10-slim",
)
def evaluate_churn_model(
    X_test: Input[Dataset],  # Renamed back
    y_test: Input[Dataset],  # Renamed back
    model_input: Input[Model], # Keep this name for the input model artifact
    metrics: Output[Metrics]   # Use KFP Metrics artifact
):
    """Evaluates the trained model and logs metrics."""
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import joblib

    print(f"Loading test data: X from {X_test.path}, y from {y_test.path}")
    # Use the input parameter names to load data
    X_test_df = pd.read_csv(X_test.path)
    y_test_df = pd.read_csv(y_test.path)
    y_test_series = y_test_df['Exited'] # Extract the Series

    print(f"Loading model from: {model_input.path}")
    # Load the model using joblib
    rf_model = joblib.load(model_input.path)

    print("Evaluating model performance...")
    # Make predictions on the test set
    y_pred = rf_model.predict(X_test_df)

    # Calculate evaluation metrics
    acc = accuracy_score(y_test_series, y_pred)
    precision = precision_score(y_test_series, y_pred)
    recall = recall_score(y_test_series, y_pred)
    f1 = f1_score(y_test_series, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Log metrics to KFP UI using Output[Metrics]
    metrics.log_metric("accuracy", round(acc, 4))
    metrics.log_metric("precision", round(precision, 4))
    metrics.log_metric("recall", round(recall, 4))
    metrics.log_metric("f1_score", round(f1, 4))


# Pipeline Definition
@pipeline(name="customer-churn-pipeline-assignment10", description="Pipeline for Customer Churn Prediction")
def customer_churn_pipeline(
    drop_missing_vals: bool = False, # Defaulting to False as train_model.py didn't drop NAs
    test_size: float = 0.2,
    random_state: int = 3, # Match random_state used in train_model.py split
    n_estimators: int = 100, # Match n_estimators used in train_model.py
):
    """Defines the workflow of the customer churn prediction pipeline."""

    # Step 1: Load and preprocess data
    load_op = load_churn_data(
        drop_missing_vals=drop_missing_vals
    )

    # Step 2: Split data into train and test sets
    split_op = train_test_split_churn(
        input_churn_dataset=load_op.outputs['churn_dataset'],
        test_size=test_size,
        random_state=random_state
    )

    # Step 3: Train the Random Forest model
    # Use the corrected output keys from split_op
    train_op = train_churn_model(
        X_train=split_op.outputs['X_train'],
        y_train=split_op.outputs['y_train'],
        n_estimators=n_estimators,
        random_state=random_state # Use same random state for model consistency if needed
    )

    # Step 4: Evaluate the model
    # Use the corrected output keys from split_op
    evaluate_op = evaluate_churn_model(
        X_test=split_op.outputs['X_test'],
        y_test=split_op.outputs['y_test'],
        model_input=train_op.outputs['model_output'] # Ensure input name matches component
    )


# Compile Pipeline
if __name__ == '__main__':
    print("Compiling pipeline...")
    # Ensure you have kfp installed: pip install kfp==1.8.5 (or compatible version)
    kfp.compiler.Compiler().compile(
        pipeline_func=customer_churn_pipeline,
        package_path='customer_churn_pipeline_v1.yaml' # Output YAML file name
    )
    print("Pipeline compiled successfully to customer_churn_pipeline_v1.yaml")