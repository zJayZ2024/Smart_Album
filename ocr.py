from paddleocr import PaddleOCR
import os
import tempfile
import cv2
import numpy as np
from PIL import Image

ocr = None

def extract_text_from_image(pil_image):

    global ocr
    if ocr is None:
        ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # 支持中英文、数字识别

    """从图像中提取文本并在图像上标注"""
    # 将PIL图像转换为OpenCV格式
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    try:
        # 执行OCR识别
        result = ocr.ocr(cv_image, cls=True)
        
        # 提取识别的文本并绘制标注
        extracted_text = []
        if result is not None and len(result) > 0:  # 确保结果不为空
            for line in result[0]:  # PaddleOCR返回的是二维列表
                # 获取文本框坐标
                box = np.array(line[0], dtype=np.int32).reshape((-1, 1, 2))
                # 获取文本内容和置信度
                text = line[1][0]
                confidence = line[1][1]
                
                if confidence > 0.5:
                    # 绘制文本框
                    cv2.polylines(cv_image, [box], True, (0, 255, 0), 2)
                    # 添加文本标签
                    extracted_text.append(text)
        
        # 将OpenCV图像转回PIL格式
        annotated_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        return {
            'texts': extracted_text,
            'image': annotated_image
        }
        
    except Exception as e:
        print(f"OCR处理错误: {str(e)}")
        return {
            'texts': [],
            'image': pil_image  # 发生错误时返回原图
        }

# def extract_text_from_image(pil_image):
#     """从图像中提取文本"""
#     # 创建临时文件
#     temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
#     temp_path = temp.name
#     temp.close()  # 关闭文件句柄，释放锁定
    
#     try:
#         # 保存图像到临时文件
#         pil_image.save(temp_path)
        
#         # 执行OCR识别
#         result = ocr.ocr(temp_path, cls=True)
        
#         # 提取识别的文本
#         extracted_text = []
#         for line in result:
#             for item in line:
#                 text = item[1][0]  # 获取识别的文本
#                 confidence = item[1][1]  # 获取置信度
#                 if confidence > 0.5:  # 只保留置信度较高的结果
#                     extracted_text.append(text)
        
#         return extracted_text
#     finally:
#         # 清理临时文件
#         if os.path.exists(temp_path):
#             os.remove(temp_path)