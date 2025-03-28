import os
import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)

# Dataset directories
DATASET_DIR = 'D:/Project/carrot_dataset'
HEALTHY_DIR = os.path.join(DATASET_DIR, 'healthy')
SPOILED_DIR = os.path.join(DATASET_DIR, 'spoiled')

# Ensure upload directories exist
os.makedirs('uploads/healthy', exist_ok=True)
os.makedirs('uploads/spoiled', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

# Image Data Generator for training
train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_data_gen.flow_from_directory(
    DATASET_DIR,
    target_size=(150, 150),
    batch_size=4,
    class_mode='binary'
)

# Build the model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)

# Save the model
model.save('carrot_classifier_model.h5')

# Load the saved model
model = tf.keras.models.load_model('carrot_classifier_model.h5')

# Function to classify carrot image
def classify_carrot(img):
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction[0][0]

# Function to calculate quality scores
def calculate_quality_score(img):
    img = img.resize((150, 150))
    img_array = np.array(img)
    avg_color = np.mean(img_array, axis=(0, 1))
    color_score = np.linalg.norm(avg_color - np.array([255, 102, 0]))
    aspect_ratio = img_array.shape[1] / img_array.shape[0]
    shape_score = 1 - abs(aspect_ratio - 1.5)
    size_score = img_array.size / (150 * 150 * 3)
    total_score = (1/color_score) + (1/shape_score) + (1/size_score)
    return color_score, shape_score, size_score, total_score

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify_carrot', methods=['POST'])
def classify():
    if 'carrotImage' not in request.files:
        return "No file uploaded.", 400

    file = request.files['carrotImage']
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    img = Image.open(img_path)
    prediction = classify_carrot(img)
    classification = "healthy" if prediction < 0.5 else "spoiled"

    color_score, shape_score, size_score, total_score = calculate_quality_score(img)

    if classification == "healthy":
        actual_img_path = os.path.join('uploads', file.filename)
    else:
        os.rename(img_path, os.path.join('uploads/spoiled', file.filename))
        actual_img_path = os.path.join('uploads/spoiled', file.filename)

    return render_template('result.html', classification=classification, img_path=actual_img_path,
                           color_score=color_score, shape_score=shape_score, 
                           size_score=size_score, total_score=total_score)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/feedback', methods=['POST'])
def feedback(): 
    img_path = request.form['img_path']
    feedback = request.form['feedback']

    if feedback == "correct":
        return "Thank you for your feedback! The image has been recorded."

    if feedback == "healthy":
        new_path = os.path.join('uploads/healthy', os.path.basename(img_path))
    else:
        new_path = os.path.join('uploads/spoiled', os.path.basename(img_path))

    os.rename(img_path, new_path)

    return "Thank you for your feedback! The image has been recorded."

# Run the Flask app
if __name__ == '__main__':
    # Host set to 127.0.0.1 and debug disabled for Electron
    app.run(host="127.0.0.1", port=5000, debug=False)
