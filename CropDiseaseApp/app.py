from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('crop_disease_model.h5')

# Upload folder setup
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class names
class_names = ['Healthy Cotton', 'Cotton Disease']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        
        if file.filename == '':
            return 'No selected file'
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            img = image.load_img(filepath, target_size=(64, 64))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)
            result = class_names[predicted_class[0]]
            
            return render_template('result.html' , result=result, user_image=filepath)
    
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)