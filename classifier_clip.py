# # classifier_clip.py
import torch
import clip
from PIL import Image

# class CLIPClassifier:
#     def __init__(self):
#         self.model, self.preprocess = clip.load("ViT-B/32")
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model.to(self.device)
#
#         # 可自定义中文类别
#         self.categories = ["人物", "动物", "风景", "美食","植物"]
#         self.prompts = clip.tokenize(["a person", "an animal", "a landscape", "delicious food","green plants or flowers or "grass" or "field""]).to(self.device)
#
#     def classify(self, image_path):
#         image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             image_features = self.model.encode_image(image)
#             text_features = self.model.encode_text(self.prompts)
#             probs = (image_features @ text_features.T).softmax(dim=-1)
#             return self.categories[probs.argmax().item()]

# classifier_clip.py
#
# import torch
# import clip
# from PIL import Image
#
# class CLIPClassifier:
#     def __init__(self):
#         self.model, self.preprocess = clip.load("ViT-B/32")
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model.to(self.device)
#
#         # 类别标签
#         self.categories = ["人物", "动物", "风景", "美食", "植物"]
#
#         # 多提示语支持（每类可多个）
#         self.category_prompts = {
#             "人物": ["a person"],
#             "动物": ["an animal"],
#             "风景": ["a landscape"],
#             "美食": ["delicious food"],
#             "植物": [
#                 "green plants",
#                 "flowers",
#                 "a garden",
#                 "a tree",
#                 "leaves",
#                 "a potted plant",
#             ]
#         }
#
#         # 展平提示并记住索引
#         self.prompts = []
#         self.prompt_to_category = []
#         for cat, phrases in self.category_prompts.items():
#             self.prompts.extend(phrases)
#             self.prompt_to_category.extend([cat] * len(phrases))
#
#         self.tokenized_prompts = clip.tokenize(self.prompts).to(self.device)
#
#     def classify(self, image_path):
#         image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             image_features = self.model.encode_image(image)
#             text_features = self.model.encode_text(self.tokenized_prompts)
#             probs = (image_features @ text_features.T).softmax(dim=-1)
#             best_index = probs.argmax().item()
#             return self.prompt_to_category[best_index]


class CLIPClassifier:
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-B/32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # 多个提示词映射
        self.categories = ["人物", "动物", "风景", "美食", "植物"]
        self.category_prompt_slices = [1, 1, 1, 1, 6]  # 每个类别对应的提示词数量

        self.prompts = [
            # 人物（1）
            "a person",
            # 动物（2）
            "an animal",
            # 风景（2）
            "a landscape",
            # 美食（2）
            "delicious food",
            # 植物（4）
            "green plants", "flowers", "a garden", "a bush with leaves","a potted plant","a tree"
        ]
        self.tokenized_prompts = clip.tokenize(self.prompts).to(self.device)

    def classify(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(self.tokenized_prompts)
            probs = (image_features @ text_features.T).softmax(dim=-1).squeeze()

        # 分组取每组的最大概率，再选整体最大组
        group_scores = []
        idx = 0
        for count in self.category_prompt_slices:
            group_scores.append(probs[idx:idx+count].max().item())
            idx += count

        best_category_idx = group_scores.index(max(group_scores))
        return self.categories[best_category_idx]
