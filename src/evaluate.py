"""
Evaluation utilities: predictions, metrics.
"""
import numpy as np
import torch


@torch.no_grad()
def get_all_predictions(model, loader, device):
    """Run model on entire loader, return (true_labels, predicted_labels)."""
    model.eval()
    all_preds = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)