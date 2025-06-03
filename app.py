from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import logging  # 添加此行
from classify import classify_uploaded_image
from ocr import extract_text_from_image
from io import BytesIO
import base64
from image_processor import ImageResizer

# 设置日志级别为 WARNING，这样就不会显示 DEBUG 和 INFO 级别的日志
logging.getLogger("ppocr").setLevel(logging.WARNING)

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/ocr', methods=['GET'])
def ocr_get():
    return render_template("ocr.html")

@app.route('/ocr', methods=['POST'])
def ocr_process():
    if 'image' not in request.files:
        return jsonify({"error": "没有上传图片"}), 400
    
    image_file = request.files['image']
    image = Image.open(image_file.stream).convert("RGB")
    
    # 执行OCR识别并获取标注后的图片
    result = extract_text_from_image(image)
    
    # 将标注后的图片转换为base64
    buffered = BytesIO()
    result['image'].save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return jsonify({
        "success": True,
        "texts": result['texts'],
        "image": f"data:image/jpeg;base64,{img_str}"
    })


@app.route('/classify', methods=['GET'])
def classify_get():
    return render_template("classify.html")

@app.route('/classify', methods=['POST'])
def classify():
    images = request.files.getlist('images')
    results = []

    for image_file in images:
        image = Image.open(image_file.stream).convert("RGB")
        category = classify_uploaded_image(image)
        results.append(category)

    return jsonify({"categories": results})

resizer = ImageResizer()

@app.route('/resize', methods=['GET'])
def resize_get():
    return render_template("resize.html")

@app.route('/resize', methods=['POST'])
def resize_process():
    if 'image' not in request.files:
        return jsonify({"error": "没有上传图片"}), 400
    
    image_file = request.files['image']
    image = Image.open(image_file.stream).convert("RGB")
    
    try:
        resize_type = request.form.get('type', 'pixels')
        
        if resize_type == 'pixels':
            width = int(request.form.get('width', 800))
            height = int(request.form.get('height', 600))
            resized = resizer.resize_by_pixels(image, width, height)
        else:
            ratio = float(request.form.get('ratio', 1.0))
            resized = resizer.resize_by_ratio(image, ratio)
        
        return jsonify({
            "success": True,
            "image": resizer.to_base64(resized)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
