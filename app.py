import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import pandas as pd
from datetime import datetime
import typer
from typing import Optional
from pathlib import Path
import random
import shutil

app = typer.Typer()

# Initialize globals
try:
    base_network = load_model('base_network.h5', compile=False)
    typer.echo("Base network loaded successfully.")
except FileNotFoundError:
    typer.echo("Error: base_network.h5 not found. Please ensure it is in the project root.")
    raise SystemExit(1)

detector = MTCNN()
face_database_path = "face_database"
embedding_database_file = "embedding_database.npy"
os.makedirs(face_database_path, exist_ok=True)

# Load or initialize embedding database
embedding_database = {}
if os.path.exists(embedding_database_file):
    try:
        embedding_database = np.load(embedding_database_file, allow_pickle=True).item()
        typer.echo(f"Loaded embedding database with {len(embedding_database)} individuals.")
    except (EOFError, ValueError):
        typer.echo("Warning: embedding_database.npy is empty or corrupted. Initializing empty database.")
        embedding_database = {}
else:
    typer.echo("No embedding database found. Initializing empty database.")

# Helper functions
def load_and_preprocess_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((100, 100, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

def detect_and_crop_face(image: np.ndarray, size: tuple = (100, 100)) -> Optional[np.ndarray]:
    # Ensure image is uint8 for cvtColor
    if image.dtype != np.uint8:
        if image.max() <= 1.0:  # Image is normalized [0,1]
            image = (image * 255.0).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    if len(faces) == 0:
        return None
    face = faces[0]
    x, y, w, h = face['box']
    x, y = max(0, x), max(0, y)
    face_img = img_rgb[y:y+h, x:x+w]
    if face_img.size == 0:  # Check for empty crop
        return None
    face_img = cv2.resize(face_img, size)
    return face_img.astype(np.float32) / 255.0

def compute_cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    dot_product = np.sum(emb1 * emb2, axis=-1)
    norm1 = np.sqrt(np.sum(emb1 * emb1, axis=-1))
    norm2 = np.sqrt(np.sum(emb2 * emb2, axis=-1))
    cosine_similarity = dot_product / (norm1 * norm2 + 1e-10)
    return (1 - cosine_similarity).item()

def log_attendance(person: str, csv_path: str = "attendance.csv") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {"Timestamp": timestamp, "Person": person}
    df = pd.DataFrame([entry])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)
    return timestamp

def identify_person(image: np.ndarray, threshold: float = 0.5) -> tuple[Optional[str], Optional[float], str]:
    face_img = detect_and_crop_face(image)
    if face_img is None:
        return None, None, "no_face"
    emb = base_network.predict(np.expand_dims(face_img, axis=0), verbose=0)[0]
    min_distance = float('inf')
    identified_person = None
    for person, embeddings in embedding_database.items():
        for db_emb in embeddings:
            distance = compute_cosine_distance(emb, db_emb)
            if distance < min_distance:
                min_distance = distance
                identified_person = person
    if min_distance > threshold:
        return None, min_distance, "no_match"
    return identified_person, min_distance, "match"

def capture_images(person_name: str, output_dir: str = "temp_images", num_images: int = 5) -> None:
    # Clear temp_images folder to avoid mixing images from previous runs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        typer.echo("Error: Could not open webcam.")
        return
    
    typer.echo(f"Capturing {num_images} images for {person_name}. Press 'c' to capture, 'q' to quit.")
    captured = 0
    
    while captured < num_images:
        ret, frame = cap.read()
        if not ret:
            typer.echo("Error: Could not read frame.")
            break
        
        cv2.imshow("Capture Image", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            img_path = os.path.join(output_dir, f"{person_name}_{captured+1}.jpg")
            cv2.imwrite(img_path, frame)
            typer.echo(f"Saved image: {img_path}")
            captured += 1
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def crop_and_save_faces(input_dir: str, person_name: str, output_dir: str = face_database_path) -> bool:
    person_output_path = os.path.join(output_dir, person_name)
    os.makedirs(person_output_path, exist_ok=True)
    
    typer.echo(f"Processing images for {person_name}...")
    valid_images = False
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        face_img = detect_and_crop_face(img)
        if face_img is None:
            typer.echo(f"No face detected in {img_name}")
            continue
        output_img_path = os.path.join(person_output_path, img_name)
        cv2.imwrite(output_img_path, cv2.cvtColor(face_img * 255, cv2.COLOR_RGB2BGR))
        typer.echo(f"Saved cropped image: {output_img_path}")
        valid_images = True
    return valid_images

def add_person_to_database(person_name: str, max_images: int = 5) -> bool:
    person_path = os.path.join(face_database_path, person_name)
    if not os.path.exists(person_path):
        typer.echo(f"Error: No images found for {person_name} in {face_database_path}")
        return False
    
    images = [os.path.join(person_path, img) for img in os.listdir(person_path)]
    if not images:
        typer.echo(f"No valid images for {person_name}")
        return False
    
    images = random.sample(images, min(len(images), max_images))
    embeddings = []
    for img_path in images:
        img = load_and_preprocess_image(img_path)
        emb = base_network.predict(np.expand_dims(img, axis=0), verbose=0)[0]
        embeddings.append(emb)
    
    embedding_database[person_name] = np.array(embeddings)
    np.save(embedding_database_file, embedding_database)
    typer.echo(f"Added {person_name} with {len(embeddings)} embeddings")
    return True

# CLI Commands
@app.command()
def add_person(person_name: str, num_images: int = 5):
    """Add a new person to the attendance system using webcam."""
    capture_images(person_name, num_images=num_images)
    if crop_and_save_faces("temp_images", person_name):
        if add_person_to_database(person_name):
            typer.echo(f"Successfully added {person_name} to the database.")
        else:
            typer.echo(f"Failed to add {person_name} to the database.")
    else:
        typer.echo(f"No valid faces detected for {person_name}. Addition failed.")

@app.command()
def detect(threshold: float = 0.5):
    """Detect faces in real-time using webcam. Press 'v' to log attendance, 'q' to quit."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        typer.echo("Error: Could not open webcam.")
        return
    
    typer.echo("Press 'v' to detect and log attendance, 'q' to quit.")
    last_logged = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            typer.echo("Error: Could not read frame.")
            break
        
        cv2.imshow("Attendance System", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('v'):
            person, distance, status = identify_person(frame, threshold)
            if status == "no_face":
                typer.echo("No person detected in the camera.")
            elif status == "no_match":
                typer.echo(f"No match found, Distance: {distance:.4f}")
            elif person:
                current_time = datetime.now()
                if person not in last_logged or (current_time - last_logged[person]).total_seconds() > 60:
                    timestamp = log_attendance(person)
                    last_logged[person] = current_time
                    typer.echo(f"Logged attendance for {person} at {timestamp}, Distance: {distance:.4f}")
                else:
                    typer.echo(f"Already logged for {person} recently, Distance: {distance:.4f}")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app()