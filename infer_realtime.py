import cv2, numpy as np, mediapipe as mp, tensorflow as tf, pyttsx3, argparse
# We only need the keypoint extraction function from utils now
from utils import extract_keypoints 

parser = argparse.ArgumentParser()
# Updated to expect the new keypoint model path
parser.add_argument("--model", type=str, default="models/sign_model_mlp_saved") 
parser.add_argument("--threshold", type=float, default=0.7) 
parser.add_argument("--use_tts", action="store_true")
args = parser.parse_args()

print("🔍 Loading keypoint MLP model...")
try:
    model = tf.keras.models.load_model(args.model, compile=False) 
    print(f"✅ Model loaded. Expecting input shape: {model.input_shape}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

try:
    with open("models/classes.txt") as f:
        classes = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print("❌ 'models/classes.txt' not found. Ensure training was completed.")
    exit()

tts = pyttsx3.init() if args.use_tts else None
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("🎥 Webcam started (press 'q' to quit)")

# Set max_num_hands to 2 to detect both hands
with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            
            # --- 1. Extract Keypoints and Predict ---
            # Extracts up to 2 hands and pads the feature vector to 126
            input_data = extract_keypoints(results, max_hands=2)
            
            # Predict using the keypoint data
            preds = model.predict(input_data, verbose=0)[0]
            idx, conf = np.argmax(preds), np.max(preds)

            # --- 2. Visualization and TTS ---
            if conf > args.threshold:
                label = f"{classes[idx]} ({conf*100:.1f}%)"
                # Display the prediction at the top left
                cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                
                if tts: 
                    # Add simple debouncing for TTS
                    if not hasattr(tts, 'last_spoken') or (cv2.getTickCount() - tts.last_spoken) / cv2.getTickFrequency() > 2:
                        tts.say(classes[idx])
                        tts.runAndWait()
                        tts.last_spoken = cv2.getTickCount()

            # Draw landmarks for ALL detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
        cv2.imshow("Sign Language Detector (Keypoints)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print("👋 Session ended.")