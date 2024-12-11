import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from scipy.spatial.distance import euclidean

# Load the model and labels
model = load_model("E:\\yoga dataet\\yoga_pose_neural_network_model.h5")
LABELS = ["downdog", "goddess", "plank", "tree", "warrior2"]

# Example: Ideal landmarks for yoga poses (replace with your data)

print("Available poses:", LABELS)
chosen_pose = input("Enter the pose you want to perform: ").strip().lower()

# Ensure the chosen pose is valid
if chosen_pose not in LABELS:
    print(f"Pose '{chosen_pose}' is not available. Please choose from {LABELS}.")
    exit()

# Get the ideal landmarks for the chosen pose


# Initialize MediaPipe Pose
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

    if landmarks.sum() != 0:
        landmarks = landmarks.reshape(1, -1)
        pose_class = model.predict(landmarks)
        pose_class_index = np.argmax(pose_class)
        predicted_pose = LABELS[pose_class_index]

        pose_probability = pose_class[0][pose_class_index]*100  # Probability of the predicted pose

# Display the predicted pose and probability
        if pose_probability<0.8 or predicted_pose!=chosen_pose:
            cv2.putText(image, "No pose detected.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, f"Pose: {predicted_pose} ({pose_probability:.2f}%)", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


        
        

        # Display accuracy and ideal points
        cv2.putText(image, f"Chosen Pose: {chosen_pose.capitalize()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
       
        # Draw the ideal points on the frame
        
        # Display feedback
        if predicted_pose == chosen_pose:
            feedback = "Good alignment!"
        else:
            feedback = f"Adjust to match {chosen_pose.capitalize()} pose."

        cv2.putText(image, feedback, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    else:
        cv2.putText(image, "No pose detected.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Yoga Pose Detection", image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
