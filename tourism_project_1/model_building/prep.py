# Token comes from environment (set in Step 1 / GitHub Actions secret)
import pandas as pd, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

api  = HfApi(token=os.environ.get("HF_TOKEN"))
REPO = "SatheeshThomas/tourism-dataset_1"

df = pd.read_csv("tourism_project_1/data/tourism.csv")
print(f"Loaded: {df.shape}") 

# FIX: drop Unnamed:0 if source CSV was saved with index
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)
    print("Dropped: Unnamed: 0")

# Drop CustomerID, reset index cleanly
df.drop(columns=["CustomerID"], inplace=True)
df.reset_index(drop=True, inplace=True)

# Fill missing values
for col in df.columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)
print("Missing values handled.")

# Encode categoricals
le = LabelEncoder()
for col in ["TypeofContact","Occupation","Gender","MaritalStatus","Designation","ProductPitched"]:
    df[col] = le.fit_transform(df[col])
print("Categorical columns encoded.")

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Assert guards — crash immediately if dirty columns exist
assert "CustomerID"  not in X.columns, "CustomerID must not be in features!"
assert "Unnamed: 0"  not in X.columns, "Unnamed: 0 must not be in features!"
assert len(X.columns) == 18, f"Expected 18 features, got {len(X.columns)}! Columns: {list(X.columns)}"
print(f"✅ Features ({len(X.columns)}): {list(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")

# index=False prevents Unnamed:0 appearing when reloaded
X_train.to_csv("tourism_project_1/data/X_train.csv", index=False)
X_test.to_csv("tourism_project_1/data/X_test.csv",   index=False)
y_train.to_csv("tourism_project_1/data/y_train.csv", index=False)
y_test.to_csv("tourism_project_1/data/y_test.csv",   index=False)
print("Splits saved.")

for f in ["X_train.csv","X_test.csv","y_train.csv","y_test.csv"]:
    api.upload_file(path_or_fileobj=f"tourism_project_1/data/{f}",
                    path_in_repo=f, repo_id=REPO, repo_type="dataset")
    print(f"  Uploaded {f}")
print("✅ All splits uploaded to HF Hub.")
