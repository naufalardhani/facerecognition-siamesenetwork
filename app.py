from flask import Flask, render_template, request, jsonify
import os
import cv2
import base64
import numpy as np
from models.facerecognition import (
    detect_and_crop_face,
    identify_person,
    capture_images,
    crop_and_save_faces,
    add_person_to_database,
    log_attendance,
    base_network,
    embedding_database,
    face_database_path,
)
from datetime import datetime

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Add person route
@app.route('/add_person', methods=['POST'])
def add_person():
    person_name = request.form['person_name']
    image_data = request.form['image']
    if not person_name or not image_data:
        return jsonify({"error": "Person name and image are required!"}), 400

    # Decode Base64 image
    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Save image temporarily and process
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{person_name}.jpg")
    cv2.imwrite(temp_path, img)

    # Process the image
    if crop_and_save_faces(temp_dir, person_name):
        if add_person_to_database(person_name):
            return jsonify({"message": f"Successfully added {person_name} to the database."})
        else:
            return jsonify({"error": f"Failed to add {person_name} to the database."}), 500
    else:
        return jsonify({"error": f"No valid faces detected for {person_name}. Addition failed."}), 400

# Detect and log attendance route
@app.route('/detect', methods=['POST'])
def detect():
    image_data = request.form['image']
    if not image_data:
        return jsonify({"error": "No image data provided!"}), 400

    # Decode Base64 image
    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Process the image
    person, distance, status = identify_person(img)

    if status == "no_face":
        return jsonify({"message": "No face detected in the image."})
    elif status == "no_match":
        return jsonify({"message": "No match found.", "distance": distance})
    elif person:
        timestamp = log_attendance(person)
        return jsonify({"message": f"Person identified: {person}", "distance": distance, "timestamp": timestamp})

    return jsonify({"error": "Unknown error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=1337)