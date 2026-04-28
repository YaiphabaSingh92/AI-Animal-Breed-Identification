import torch
import timm
from torchvision import transforms
from PIL import Image
import json
import os

class BovineClassifier:
    def __init__(self):
        self.weights_path = "models/weights/Indian_bovine_finetuned_model.pth"
        self.labels_path = "models/labels/classes.json"
        self.supported_classes_path = "models/labels/supported_classes.json"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes = []
        self.supported_classes = set()
        self.mask_tensor = None
        self.transform = None
        
        self._load_labels()
        self._load_model()
        self._setup_transforms()
        
    def _load_labels(self):
        with open(self.labels_path, "r") as f:
            self.classes = json.load(f)
            
        if os.path.exists(self.supported_classes_path):
            with open(self.supported_classes_path, "r") as f:
                self.supported_classes = set(json.load(f))
        else:
            self.supported_classes = set(self.classes)
            
        # Create mask tensor (1 for supported, 0 for unsupported)
        mask = [1 if c in self.supported_classes else 0 for c in self.classes]
        self.mask_tensor = torch.tensor(mask, device=self.device)
        
    def _load_model(self):
        # We load a convnext_tiny which matches the checkpoint architecture
        self.model = timm.create_model("convnext_tiny", pretrained=False, num_classes=len(self.classes))
        
        checkpoint = torch.load(self.weights_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        
    def _setup_transforms(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # Subdue outputs completely for non-supported classes using the mask
            outputs[0, self.mask_tensor == 0] = float('-inf')
            
            # Apply softmax to get confidence percentages
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Get Top 5 Predictions
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            top_5_list = []
            for i in range(top5_prob.size(0)):
                prob_val = top5_prob[i].item() * 100
                breed_name = self.classes[top5_catid[i].item()]
                top_5_list.append({
                    "breed": breed_name.replace("_", " "),
                    "confidence": round(prob_val, 2)
                })

            pred_idx = top5_catid[0].item()
            pred_class = self.classes[pred_idx]
            confidence = top5_prob[0].item() * 100
            
            return {
                "breed": pred_class,
                "confidence": round(confidence, 2),
                "top_5": top_5_list
            }

classifier = BovineClassifier()

def get_classifier():
    return classifier
