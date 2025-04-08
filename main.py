# main.py
from detector_yolo import YOLOPersonDetector
from classifier_clip import CLIPClassifier
from helper import load_images_from_folder, ensure_output_folders, move_to_category
import os

INPUT_FOLDER = "image"
OUTPUT_FOLDER = "results"
CATEGORIES = ["人物", "动物", "风景", "美食"]

def main():
    print("🔍 初始化模型...")
    yolo = YOLOPersonDetector()
    clip_model = CLIPClassifier()
    ensure_output_folders(OUTPUT_FOLDER, CATEGORIES)

    image_paths = load_images_from_folder(INPUT_FOLDER)
    print(f"📸 共检测到 {len(image_paths)} 张图片")

    for path in image_paths:
        print(f"识别中：{path}")
        try:
            if yolo.detect_person(path):
                category = "人物"
            else:
                category = clip_model.classify(path)
            print(f"➡️ 分类为：{category}")
            move_to_category(path, category, OUTPUT_FOLDER)
        except Exception as e:
            print(f"❌ 错误：{e}，跳过 {path}")

if __name__ == "__main__":
    main()
