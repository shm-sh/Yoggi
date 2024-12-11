import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import joblib


model = load_model("E:\yoga dataet\yoga_pose_neural_network_model.h5")

LABELS =["downdog", "goddess", "plank", "tree", "warrior2"]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def extract_landmarks(results):
    """Extract pose landmarks as a flattened list."""
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        return np.zeros(132)  # 33 landmarks * 4 attributes

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Detect pose
    results = pose.process(image)

    # Convert back to BGR for display
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw pose landmarks on the image
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Extract landmarks
    landmarks = extract_landmarks(results)

    # Predict yoga pose
    if landmarks.sum() != 0:  # Check if landmarks are valid
        landmarks = landmarks.reshape(1, -1)  # Reshape for prediction
        pose_class = model.predict(landmarks)  # Use the trained model for prediction

        if hasattr(model, "predict_proba"):  # For Random Forest/XGBoost
            pose_class = np.argmax(pose_class)

        # Get the class index with the highest probability
        pose_class_index = np.argmax(pose_class) if isinstance(pose_class, np.ndarray) else pose_class
        pose_label = LABELS[pose_class_index]


        # Display predicted pose on the frame
        cv2.putText(image, f"Pose: {pose_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, "Pose: Unknown", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Yoga Pose Detection", image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()