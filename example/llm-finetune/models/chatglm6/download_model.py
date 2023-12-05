
from huggingface_hub import snapshot_download

try:
    from utils import BASE_MODEL_DIR
except ImportError:
    from .utils import BASE_MODEL_DIR


BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    snapshot_download(repo_id="THUDM/chatglm6b", local_dir=BASE_MODEL_DIR)
