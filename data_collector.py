import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os
from glob import glob
from tqdm import tqdm

# --- Configuration ---
ROOT_DATA_DIR = 'dataset' # Path to your 'dataset' folder containing 'train' and 'val'
DATA_PATH = 'keypoint_data.csv' # Output CSV file

# Initialize MediaPipe Hands
# We set static_image_mode=True for better accuracy on individual images
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils 

# Load existing data or create a new DataFrame
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded existing data. Total initial rows: {len(df)}")
else:
    # 126 features (2 hands * 21 points * 3 coords) + 1 class label
    columns = [f'feature_{i}' for i in range(126)] + ['class']
    df = pd.DataFrame(columns=columns)
    print("Created new keypoint_data.csv DataFrame.")

# --- Keypoint Extraction Logic ---

def extract_keypoints(results, max_hands=2):
    # This function is assumed to be in utils.py and remains the same
    keypoints = []
    detected_hands = results.multi_hand_landmarks or []
    for hand_landmarks in detected_hands:
        for landmark in hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
    
    expected_features = 21 * 3 * max_hands 
    if len(keypoints) < expected_features:
        keypoints.extend([0.0] * (expected_features - len(keypoints)))
    keypoints = keypoints[:expected_features]
    return np.array([keypoints], dtype=np.float32)

# --- Main Processing Loop ---

# Search for all image files in train/*/ and val/*/
image_paths = glob(os.path.join(ROOT_DATA_DIR, '**', '*', '*.jpg'), recursive=True) + \
              glob(os.path.join(ROOT_DATA_DIR, '**', '*', '*.jpeg'), recursive=True) + \
              glob(os.path.join(ROOT_DATA_DIR, '**', '*', '*.png'), recursive=True)

print(f"\nFound {len(image_paths)} images across all classes in {ROOT_DATA_DIR}.")

with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
    new_rows = []
    
    for image_path in tqdm(image_paths, desc="Processing Images"):
        
        # Determine the class name from the folder structure (e.g., 'dataset/train/A/image.jpg' -> 'A')
        class_name = os.path.basename(os.path.dirname(image_path))
        
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            continue
        
        # Convert to RGB and process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            # Extract the 126 features
            keypoints = extract_keypoints(results, max_hands=2)[0] 
            
            # Append class name
            row_data = list(keypoints) + [class_name]
            new_rows.append(row_data)

# Append all new rows at once
df_new = pd.DataFrame(new_rows, columns=df.columns)
df = pd.concat([df, df_new], ignore_index=True)

# Save the updated DataFrame
df.to_csv(DATA_PATH, index=False)
print(f"\n✅ Data processing complete. Total final rows in {DATA_PATH}: {len(df)}")