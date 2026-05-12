"""
Plant disease classifier — loads model once, predicts from PIL images.
"""
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.model import create_mobilenet


# Must match the eval transforms from training (no augmentation)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class PlantClassifier:
    """Wraps a trained plant disease model for inference."""

    def __init__(
        self,
        model_path: Path,
        classes_path: Path,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load class names (index -> human-readable label)
        with open(classes_path) as f:
            self.classes = json.load(f)

        # Build model architecture and load trained weights
        self.model = create_mobilenet(num_classes=len(self.classes))
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"PlantClassifier ready on {self.device} with {len(self.classes)} classes")

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 3) -> list[dict]:
        """Returns top-K predictions with confidence scores."""
        # Ensure RGB (PNGs can have alpha, etc.)
        image = image.convert("RGB")

        # Preprocess: transform + add batch dimension
        tensor = _transform(image).unsqueeze(0).to(self.device)

        # Forward pass
        logits = self.model(tensor)
        probabilities = F.softmax(logits, dim=1)[0]

        # Top-K
        top_probs, top_indices = probabilities.topk(top_k)
        return [
            {
                "class": self.classes[idx.item()],
                "confidence": float(prob.item()),
            }
            for prob, idx in zip(top_probs, top_indices)
        ]


# Quick smoke test (run: python -m api.inference)
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    classifier = PlantClassifier(
        model_path=project_root / "models" / "mobilenet_v3_small.pth",
        classes_path=project_root / "models" / "classes.json",
    )

    # Pick the first image from the Apple Scab class as a test
    test_dir = project_root / "data/raw/plantvillage dataset/color/Apple___Apple_scab"
    test_image_path = next(test_dir.iterdir())

    img = Image.open(test_image_path)
    predictions = classifier.predict(img)

    print(f"\nTest image: {test_image_path.name}")
    print(f"Expected class: Apple___Apple_scab\n")
    print("Top-3 predictions:")
    for pred in predictions:
        print(f"  {pred['class']}: {pred['confidence']:.4f}")