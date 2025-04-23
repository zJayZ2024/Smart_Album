# classifier_clip.py
import torch
import clip
from PIL import Image

class CLIPClassifier:
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-B/32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # 可自定义中文类别
        self.categories = ["人物", "动物", "风景", "美食","植物"]
        self.prompts = clip.tokenize(["a person", "an animal", "a landscape", "delicious food","green plants or flowers"]).to(self.device)

    def classify(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(self.prompts)
            probs = (image_features @ text_features.T).softmax(dim=-1)
            return self.categories[probs.argmax().item()]
