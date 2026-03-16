import torch
from torchvision import models


class ImageClassifier:
    def __init__(self):
        self.weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=self.weights)
        self.model.eval()
        self.preprocess = self.weights.transforms()
        self.labels = self.weights.meta["categories"]

    def predict(self, image_tensor):
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        confidence, predicted_idx = torch.max(probabilities, dim=0)

        return {
            "prediction": self.labels[predicted_idx.item()],
            "confidence": round(confidence.item(), 4),
            "top_5": self._top_5(probabilities),
        }

    def _top_5(self, probabilities):
        top_probs, top_idxs = torch.topk(probabilities, 5)
        results = []

        for prob, idx in zip(top_probs, top_idxs):
            results.append(
                {
                    "label": self.labels[idx.item()],
                    "confidence": round(prob.item(), 4),
                }
            )

        return results