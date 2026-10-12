"""Download the cited public benchmark dataset used for this experiment."""
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / ".datasets" / "Mayank14_Multi_Class_Waste_Image_Classification_Dataset"

snapshot_download(
    repo_id="Mayank14/Multi-Class-Waste-Image-Classification-Dataset",
    repo_type="dataset",
    local_dir=TARGET,
)
print(TARGET)
