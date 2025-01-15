import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
import logging
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """Loads the Wisconsin Breast Cancer dataset from scikit-learn.

    Returns:
        pandas.DataFrame: Loaded data as Pandas dataframe.
    """
    logging.info("Loading data")
    breast_cancer = load_breast_cancer()
    df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    df['target'] = breast_cancer.target
    logging.info("Data loaded successfully")
    return df


def preprocess_data(df, test_size=0.2, random_state=42):
    """Preprocesses the data by splitting it into training and test sets, and then scales it.

    Args:
        df (pandas.DataFrame): Input dataframe.
        test_size (float, optional): Size of the test split. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: X_train, X_test, y_train, y_test.
    """
    logging.info("Preprocessing data")
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    logging.info("Data preprocessing complete")

    return X_train, X_test, y_train, y_test


def train_and_evaluate_model(X_train, X_test, y_train, y_test, params = None):
  """Trains a SVM model, evaluates it and prints result.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        params (dict, optional): Hyperparameters for GridSearchCV. Defaults to None.
    Returns:
       sklearn.model: trained model
    """
  if params is None:
        logging.info("No hyperparameter settings used.")
        model = SVC(probability=True)
        model.fit(X_train,y_train)
  else:
      logging.info("Performing hyperparameter optimization")
      grid = GridSearchCV(SVC(probability=True),param_grid=params)
      grid.fit(X_train, y_train)
      model = grid.best_estimator_
      logging.info(f"Optimal parameters {grid.best_params_}")


  y_pred = model.predict(X_test)
  y_pred_prob = model.predict_proba(X_test)[:, 1] # Probability of class 1 (malignant)

  # Evaluate the model
  accuracy = accuracy_score(y_test, y_pred)
  logging.info(f"Accuracy: {accuracy}")

  report = classification_report(y_test, y_pred)
  logging.info(f"Classification report: \n {report}")

  auc = roc_auc_score(y_test, y_pred_prob)
  logging.info(f"AUC score: {auc}")

  return model, y_pred_prob

def visualize_results(y_test, y_pred_prob):
  """Visualizes the ROC curve.

    Args:
        y_test: Test labels.
        y_pred_prob: Predicted probabilities.
  """

  fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
  plt.plot(fpr, tpr, color="darkorange", label = "ROC curve (area=%0.2f)" % roc_auc_score(y_test,y_pred_prob))
  plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("ROC Curve")
  plt.legend()
  plt.show()


def main():
    """Main function to run the SVM model."""

    # Load config
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = None

    if config is not None:
        test_size = config.get('test_size', 0.2)
        random_state = config.get('random_state', 42)
        grid_params = config.get('grid_params', None)
    else:
        test_size = 0.2
        random_state = 42
        grid_params = None
        logging.warning("Config not provided. Using default settings.")

    # Load, Preprocess Data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df, test_size=test_size, random_state=random_state)

    # Train and Evaluate Model
    model, y_pred_prob = train_and_evaluate_model(X_train, X_test, y_train, y_test, params=grid_params)

    # Visualize Results
    visualize_results(y_test, y_pred_prob)


if __name__ == "__main__":
    main()