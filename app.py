from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
from classify import classify_uploaded_image

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/classify', methods=['POST'])
def classify():
    images = request.files.getlist('images')
    results = []

    for image_file in images:
        image = Image.open(image_file.stream).convert("RGB")
        category = classify_uploaded_image(image)
        results.append(category)

    return jsonify({"categories": results})

if __name__ == '__main__':
    app.run(debug=True)
