# Token comes from environment (set in Step 1 / GitHub Actions secret)
import pandas as pd, os, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

api = HfApi(token=os.environ.get("HF_TOKEN"))

print("Loading data from HF Hub...")
X_train = pd.read_csv("hf://datasets/SatheeshThomas/tourism-dataset_1/X_train.csv")
X_test  = pd.read_csv("hf://datasets/SatheeshThomas/tourism-dataset_1/X_test.csv")
y_train = pd.read_csv("hf://datasets/SatheeshThomas/tourism-dataset_1/y_train.csv").squeeze()
y_test  = pd.read_csv("hf://datasets/SatheeshThomas/tourism-dataset_1/y_test.csv").squeeze()
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

assert "Unnamed: 0" not in X_train.columns, "Unnamed:0 in features!"
assert "CustomerID" not in X_train.columns, "CustomerID in features!"
assert len(X_train.columns) == 18, f"Expected 18 features, got {len(X_train.columns)}!"
print("✅ Feature check passed.")

class_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
model_pipeline = make_pipeline(
    StandardScaler(),
    xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42,
                      eval_metric="logloss"),
)
param_grid = {
    "xgbclassifier__n_estimators":     [50, 100],
    "xgbclassifier__max_depth":        [3, 4],
    "xgbclassifier__learning_rate":    [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.5, 0.6],
    "xgbclassifier__reg_lambda":       [0.4, 0.5],
}
gs = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)
print(f"Best params: {gs.best_params_}")

best_model = gs.best_estimator_
y_pred = (best_model.predict_proba(X_test)[:,1] >= 0.45).astype(int)
print(classification_report(y_test, y_pred))

joblib.dump(best_model, "best-tourism-model-v2.joblib")
REPO_ID = "SatheeshThomas/tourism-prj-model-1"
try:
    api.repo_info(repo_id=REPO_ID, repo_type="model")
except RepositoryNotFoundError:
    api.create_repo(repo_id=REPO_ID, repo_type="model", private=False)

api.upload_file(path_or_fileobj="best-tourism-model-v2.joblib",
                path_in_repo="best-tourism-model-v2.joblib",
                repo_id=REPO_ID, repo_type="model")
print(f"✅ Model uploaded: {REPO_ID}")
