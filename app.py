import cv2
import numpy as np
import os
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
SKETCH_FOLDER = os.path.join(UPLOAD_FOLDER, 'sketches')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SKETCH_FOLDER'] = SKETCH_FOLDER

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SKETCH_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    else:
        print("Image loaded Successfully")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def convert_to_grayscale(image_rgb):
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

def invert_image(image_gray):
    return 255 - image_gray

def blur_image(image, kernel_size=(21, 21)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def dodge_blend(greyscale, inverted_blur):
    greyscale_float = greyscale.astype(np.float32)
    inverted_blur_safe = inverted_blur.astype(np.float32) + 1e-5
    divisor = 255 - inverted_blur_safe
    blend = cv2.divide(greyscale_float, divisor, scale=256.0)
    return np.clip(blend, 0, 255).astype(np.uint8)

def save_image(output_path, image):
    cv2.imwrite(output_path, image)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(input_path)
        print(f"File saved at: {input_path}")
        print(f"Exists: {os.path.exists(input_path)}")

        try:
            image_rgb = read_image(input_path)
            gray_image = convert_to_grayscale(image_rgb)
            inverted_image = invert_image(gray_image)
            blurred_image = blur_image(inverted_image)
            sketch = dodge_blend(gray_image, blurred_image)

            sketch_filename = f"sketch_{unique_filename}"
            output_path = os.path.join(app.config['SKETCH_FOLDER'], sketch_filename)
            save_image(output_path, sketch)
            
            return jsonify({'sketch_url': f'/sketches/{sketch_filename}'})
        except Exception as e:
            print(f"An error occurred: {e}")
            return jsonify({'error': 'Failed to process the image.'}), 500
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/sketches/<filename>')
def send_sketch(filename):
    return send_from_directory(app.config['SKETCH_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=10000)

