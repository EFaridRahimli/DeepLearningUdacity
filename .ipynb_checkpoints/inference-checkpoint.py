import os
import json
import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def model_fn(model_dir):
    model = models.resnet18(pretrained=False)
    
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 133)
    )
    
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"), map_location=torch.device("cpu")))
    model.eval()
    return model


def input_fn(request_body, content_type):
    if content_type == "application/x-image":
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    else:
        raise Exception(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    with torch.no_grad():
        outputs = model(input_data)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()


def output_fn(prediction, content_type="application/json"):
    if content_type == "application/json":
        return json.dumps({"predicted_class_index": prediction})
    else:
        raise Exception({content_type}")