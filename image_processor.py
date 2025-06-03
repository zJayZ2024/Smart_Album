from PIL import Image
import os
from io import BytesIO
import base64

class ImageResizer:
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.gif']

    def resize_by_pixels(self, image, width, height):
        """按像素调整图像大小"""
        try:
            resized_image = image.resize((width, height), Image.Resampling.LANCZOS)
            return resized_image
        except Exception as e:
            raise Exception(f"调整图像大小时出错：{str(e)}")

    def resize_by_ratio(self, image, ratio):
        """按比例调整图像大小"""
        try:
            new_width = int(image.width * ratio)
            new_height = int(image.height * ratio)
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return resized_image
        except Exception as e:
            raise Exception(f"调整图像大小时出错：{str(e)}")

    def to_base64(self, image):
        """将PIL图像转换为base64字符串"""
        try:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            raise Exception(f"转换图像格式时出错：{str(e)}")