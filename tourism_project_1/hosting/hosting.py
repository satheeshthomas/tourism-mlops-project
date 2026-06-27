# Token comes from environment (set in Step 1 / GitHub Actions secret)
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
import os

api   = HfApi(token=os.environ.get("HF_TOKEN"))
SPACE = "SatheeshThomas/tourism-project-1"

try:
    api.repo_info(repo_id=SPACE, repo_type="space")
    print(f"Space {SPACE} already exists.")
except RepositoryNotFoundError:
    api.create_repo(repo_id=SPACE, repo_type="space", space_sdk="docker", private=False)
    print(f"Space {SPACE} created.")

try:
    api.delete_file(path_in_repo="streamlit_app.py", repo_id=SPACE, repo_type="space")
    print("Deleted default streamlit_app.py.")
except Exception:
    pass

for filename in ["Dockerfile","app.py","requirements.txt"]:
    api.upload_file(
        path_or_fileobj=f"tourism_project_1/deployment/{filename}",
        path_in_repo=filename, repo_id=SPACE, repo_type="space")
    print(f"  Uploaded {filename}")

print(f"\n✅ Deployment complete: https://huggingface.co/spaces/{SPACE}")
print("Space rebuilds in ~3 minutes.")
