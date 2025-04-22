from detector_yolo import YOLOPersonDetector
from classifier_clip import CLIPClassifier
from PIL import Image
import tempfile
import os

# 模型初始化（只加载一次）
yolo = YOLOPersonDetector()
clip_model = CLIPClassifier()

def classify_uploaded_image(pil_image):
    # 创建临时文件但不在 with 中写入内容，防止句柄锁定
    temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    temp_path = temp.name
    temp.close()  # 关闭文件句柄，释放锁定

    try:
        # 保存图像到临时文件
        pil_image.save(temp_path)

        # 使用 YOLO 检测人
        if yolo.detect_person(temp_path):
            return "人物"
        else:
            return clip_model.classify(temp_path)
    finally:
        # 无论如何都清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
