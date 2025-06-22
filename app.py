from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import seaborn as sns
import cv2
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'Uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = tf.keras.models.load_model(r"C:\Users\premp\Downloads\Semantic_Segmentation_For_Autonomous_Vehicles-main\image_segmentation_model.h5")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def give_color_to_seg_img(seg, n_classes= 13):
    
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    colors = sns.color_palette("hls", n_classes)
    
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))

    return(seg_img)



def predict_and_save_image(file):
    # Load image and resize to model input size (256x256)
    image = Image.open(file.stream).convert('RGB')
    image = image.resize((256, 256))  # Resize to model input size
    image_np = np.array(image) / 255.0  # Normalize if needed

    # Predict using model
    pred = model.predict(np.expand_dims(image_np, axis=0))  # Add batch dimension

    # Convert prediction to segmentation mask
    prediction_np = (give_color_to_seg_img(np.argmax(pred[0], axis=-1)) * 255).astype(np.uint8)

    # Resize original image back to original size for overlay (optional)
    original_image = np.array(Image.open(file.stream).convert('RGB'))  # Reload to get original size
    original_image = cv2.resize(original_image, (prediction_np.shape[1], prediction_np.shape[0]))

    # Apply alpha blending
    alpha = 0.7
    overlay = cv2.addWeighted(original_image, alpha, prediction_np, 1 - alpha, 0)

    # Save output image
    base_filename = os.path.splitext(file.filename)[0]
    filename = base_filename + '_predicted.jpg'
    output_path = os.path.join('static', app.config['UPLOAD_FOLDER'], filename)

    cv2.imwrite(output_path, overlay)
    print("Written:", output_path)

    return filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact_us.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Predict image and save the predicted image
        predicted_filename = predict_and_save_image(file)
        return redirect(url_for('display_image', filename=predicted_filename))
    else:
        return "Invalid file format!"

@app.route('/Uploads/<filename>')
def display_image(filename):
    return render_template('result.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
