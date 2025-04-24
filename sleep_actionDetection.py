# uses pre-trained MediaPipe models
# Enhanced eye closure detection for more reliable sleep detection

import cv2
import numpy as np
import mediapipe as mp
import time
from google.colab import files
from IPython.display import display, HTML
import os

# Initialize MediaPipe Pose and Face Mesh
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection

# Initialize the models
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, 
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
                                  
# Face detection for more reliable face finding
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Define specific eye landmarks indices for MediaPipe Face Mesh
# These points represent the contours of the eyes more precisely
LEFT_EYE = [
    # Upper eyelid
    159, 158, 157, 173, 133,
    # Lower eyelid
    145, 144, 163, 7, 33
]

RIGHT_EYE = [
    # Upper eyelid
    384, 385, 386, 387, 388,
    # Lower eyelid
    249, 373, 374, 380, 381
]

# Specific points for vertical measurement
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

# Constants for eye aspect ratio calculation
EYE_AR_THRESHOLD = 0.17  # Lower threshold for stricter detection of eye closure
EYE_AR_CONSEC_FRAMES = 3  # Fewer consecutive frames required to detect sleep

# For debugging and visualization
DRAW_EYE_LANDMARKS = True
debug_info = {}

# Tracking for eye closure
eye_closed_frames = 0
is_sleeping = False

# Angle calculation function
def calculate_angle(a, b, c):
    """Calculate the angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def eye_aspect_ratio(landmarks, eye_indices, top_idx, bottom_idx, left_idx, right_idx):
    """
    Calculate eye aspect ratio using the formula:
    EAR = (vertical distance) / (horizontal distance)
    """
    if not landmarks:
        return 1.0  # Default to open eye if no landmarks
    
    # Get coordinates
    top = np.array([landmarks[top_idx].x, landmarks[top_idx].y])
    bottom = np.array([landmarks[bottom_idx].x, landmarks[bottom_idx].y])
    left = np.array([landmarks[left_idx].x, landmarks[left_idx].y])
    right = np.array([landmarks[right_idx].x, landmarks[right_idx].y])
    
    # Calculate vertical and horizontal distances
    vertical_dist = np.linalg.norm(top - bottom)
    horizontal_dist = np.linalg.norm(left - right)
    
    # Calculate EAR
    ear = vertical_dist / (horizontal_dist + 1e-6)
    
    return ear

def detect_eye_closure(face_landmarks, image_shape):
    """
    Enhanced detection of eye closure using precise landmarks
    Returns True if eyes are closed, False otherwise, and debug information
    """
    global debug_info
    
    if not face_landmarks:
        return False, {}
    
    height, width = image_shape[:2]
    landmarks = face_landmarks.landmark
    
    # Calculate eye aspect ratios
    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
    
    # Average of both eyes
    avg_ear = (left_ear + right_ear) / 2.0
    
    # Store debug info
    debug_info = {
        "left_ear": left_ear,
        "right_ear": right_ear,
        "avg_ear": avg_ear,
        "threshold": EYE_AR_THRESHOLD,
        "eye_points": {
            "left_top": (int(landmarks[LEFT_EYE_TOP].x * width), int(landmarks[LEFT_EYE_TOP].y * height)),
            "left_bottom": (int(landmarks[LEFT_EYE_BOTTOM].x * width), int(landmarks[LEFT_EYE_BOTTOM].y * height)),
            "right_top": (int(landmarks[RIGHT_EYE_TOP].x * width), int(landmarks[RIGHT_EYE_TOP].y * height)),
            "right_bottom": (int(landmarks[RIGHT_EYE_BOTTOM].x * width), int(landmarks[RIGHT_EYE_BOTTOM].y * height))
        }
    }
    
    # Return True if eyes are closed (ratio below threshold)
    return avg_ear < EYE_AR_THRESHOLD, debug_info

# Action detection function
def detect_actions(pose_landmarks, face_landmarks=None, image_shape=None):
    """Detect multiple actions based on pose and face landmarks"""
    global eye_closed_frames, is_sleeping
    
    actions = []
    
    # First check if eyes are closed (sleeping)
    eyes_closed = False
    if face_landmarks:
        eyes_closed, eye_debug = detect_eye_closure(face_landmarks, image_shape)
        if eyes_closed:
            eye_closed_frames += 1
            if eye_closed_frames >= EYE_AR_CONSEC_FRAMES:
                is_sleeping = True
                actions.append("Sleeping")
        else:
            eye_closed_frames = max(0, eye_closed_frames - 1)  # Gradual decrease to prevent flickering
            if eye_closed_frames < EYE_AR_CONSEC_FRAMES // 2:  # Clear sleeping status if frames below half threshold
                is_sleeping = False
    
    # Get key landmark coordinates from pose
    if pose_landmarks:
        # Convert landmarks to numpy array for easier calculation
        lm_pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_landmarks.landmark])
        
        # Get key points
        nose = lm_pose[mp_pose.PoseLandmark.NOSE.value][:2]
        left_shoulder = lm_pose[mp_pose.PoseLandmark.LEFT_SHOULDER.value][:2]
        right_shoulder = lm_pose[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][:2]
        left_elbow = lm_pose[mp_pose.PoseLandmark.LEFT_ELBOW.value][:2]
        right_elbow = lm_pose[mp_pose.PoseLandmark.RIGHT_ELBOW.value][:2]
        left_wrist = lm_pose[mp_pose.PoseLandmark.LEFT_WRIST.value][:2]
        right_wrist = lm_pose[mp_pose.PoseLandmark.RIGHT_WRIST.value][:2]
        left_hip = lm_pose[mp_pose.PoseLandmark.LEFT_HIP.value][:2]
        right_hip = lm_pose[mp_pose.PoseLandmark.RIGHT_HIP.value][:2]
        left_knee = lm_pose[mp_pose.PoseLandmark.LEFT_KNEE.value][:2]
        right_knee = lm_pose[mp_pose.PoseLandmark.RIGHT_KNEE.value][:2]
        left_ankle = lm_pose[mp_pose.PoseLandmark.LEFT_ANKLE.value][:2]
        right_ankle = lm_pose[mp_pose.PoseLandmark.RIGHT_ANKLE.value][:2]
        
        # Calculate angles
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
        right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
        
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        # Check if sitting
        # When sitting: knees are bent, ankles below knees, hips angle ~90 degrees
        if (left_knee_angle < 130 and right_knee_angle < 130) and \
           (left_hip_angle < 140 and right_hip_angle < 140) and \
           (left_ankle[1] > left_knee[1] and right_ankle[1] > right_knee[1]):
            actions.append("Sitting")
        
        # Check if standing
        elif (left_knee_angle > 160 and right_knee_angle > 160) and \
             (left_hip_angle > 160 and right_hip_angle > 160):
            actions.append("Standing")
        
        # Check if lying down
        # Check if the body is horizontal
        spine_vertical = abs(left_shoulder[1] - left_hip[1]) + abs(right_shoulder[1] - right_hip[1])
        spine_horizontal = abs(left_shoulder[0] - left_hip[0]) + abs(right_shoulder[0] - right_hip[0])
        if spine_horizontal > spine_vertical:
            actions.append("Lying Down")
        
        # Check if hands are raised
        if (left_wrist[1] < left_shoulder[1] and right_wrist[1] < right_shoulder[1] and 
            left_shoulder_angle > 120 and right_shoulder_angle > 120):
            actions.append("Hands Raised")
        
        # Check for T-pose
        if (left_shoulder_angle > 80 and left_shoulder_angle < 110 and 
            right_shoulder_angle > 80 and right_shoulder_angle < 110 and
            left_elbow_angle > 150 and right_elbow_angle > 150):
            actions.append("T-Pose")
        
        # Check for squatting
        if (left_hip_angle < 120 and right_hip_angle < 120 and
            left_knee_angle < 120 and right_knee_angle < 120 and
            left_shoulder[1] < left_hip[1] + 0.2):  # Shoulder not too far above hip
            actions.append("Squatting")
        
        # Check if arms are crossed
        if (left_wrist[0] > right_shoulder[0] and right_wrist[0] < left_shoulder[0] and
            left_elbow_angle < 110 and right_elbow_angle < 110):
            actions.append("Arms Crossed")
            
        # Check for hands-on-hips pose
        if (left_elbow_angle < 120 and right_elbow_angle < 120 and
            abs(left_wrist[0] - left_hip[0]) < 0.1 and abs(right_wrist[0] - right_hip[0]) < 0.1):
            actions.append("Hands on Hips")
        
        # Check for waving right hand
        if (right_shoulder_angle > 45 and right_elbow_angle > 100 and 
            right_wrist[1] < right_shoulder[1]):
            actions.append("Waving Right Hand")
        
        # Check for waving left hand
        if (left_shoulder_angle > 45 and left_elbow_angle > 100 and 
            left_wrist[1] < left_shoulder[1]):
            actions.append("Waving Left Hand")
    
    # If no actions were detected but sleeping is still active
    if is_sleeping and not "Sleeping" in actions:
        actions.append("Sleeping")
        
    # If no actions were detected at all
    if not actions:
        actions.append("No Action Detected")
        
    return actions, eyes_closed

def draw_debug_eye_info(image, debug_info):
    """Draw eye tracking debug information on the image"""
    if not debug_info:
        return image
        
    # Draw vertical lines for eye height measurement
    if "eye_points" in debug_info:
        points = debug_info["eye_points"]
        # Left eye
        cv2.line(image, points["left_top"], points["left_bottom"], (0, 255, 255), 1)
        # Right eye
        cv2.line(image, points["right_top"], points["right_bottom"], (0, 255, 255), 1)
        
    # Display EAR values
    y_pos = 120
    cv2.putText(image, f"Left EAR: {debug_info.get('left_ear', 0):.3f}", (10, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y_pos += 20
    cv2.putText(image, f"Right EAR: {debug_info.get('right_ear', 0):.3f}", (10, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y_pos += 20
    cv2.putText(image, f"Avg EAR: {debug_info.get('avg_ear', 0):.3f}", (10, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y_pos += 20
    cv2.putText(image, f"Threshold: {debug_info.get('threshold', 0):.3f}", (10, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y_pos += 20
    cv2.putText(image, f"Eye Closed Frames: {eye_closed_frames}/{EYE_AR_CONSEC_FRAMES}", (10, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
               
    return image

def draw_face_landmarks(image, face_landmarks):
    """Draw specific face landmarks for eyes"""
    if not face_landmarks:
        return image
        
    height, width = image.shape[:2]
    
    # Draw left eye landmarks
    for idx in LEFT_EYE:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(image, (x, y), 1, (0, 255, 255), -1)  # Yellow dots
    
    # Draw right eye landmarks
    for idx in RIGHT_EYE:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(image, (x, y), 1, (0, 255, 255), -1)  # Yellow dots
        
    return image

def process_video(video_path, output_path=None):
    """Process video file with MediaPipe and generate output video with action labels"""
    global eye_closed_frames, is_sleeping, debug_info
    
    # Reset global variables
    eye_closed_frames = 0
    is_sleeping = False
    debug_info = {}
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer if output path is provided
    if output_path is None:
        output_path = "output_" + os.path.basename(video_path)
    
    out = cv2.VideoWriter(output_path, 
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          fps, 
                          (width, height))
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {frame_count}")
    print(f"Output will be saved to: {output_path}")
    
    # Process the video frame by frame
    frame_idx = 0
    all_actions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx}/{frame_count}")
        
        # Convert the BGR image to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image with MediaPipe Pose and Face Mesh
        pose_results = pose.process(image_rgb)
        face_results = face_mesh.process(image_rgb)
        face_detection_results = face_detection.process(image_rgb)
        
        # Convert back to BGR for display and saving
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        
        # Draw pose landmarks and detect actions
        actions = ["No Pose Detected"]
        eyes_closed = False
        
        if pose_results.pose_landmarks:
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                image,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # If face detected by face detection (more reliable than mesh)
            face_detected = False
            if face_detection_results.detections:
                face_detected = True
                for detection in face_detection_results.detections:
                    # Draw bounding box around face
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * width)
                    y = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw face mesh if available
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    if DRAW_EYE_LANDMARKS:
                        image = draw_face_landmarks(image, face_landmarks)
                
                # Detect actions including eye state
                actions, eyes_closed = detect_actions(
                    pose_results.pose_landmarks, 
                    face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None,
                    image.shape
                )
                
                # Draw debug info about eye tracking
                image = draw_debug_eye_info(image, debug_info)
            else:
                # If face mesh not found but still have pose
                actions, _ = detect_actions(pose_results.pose_landmarks, None, image.shape)
            
            all_actions.append(actions)
        
        # Display actions text
        y_pos = 40
        cv2.putText(image, f"Actions:", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        y_pos += 40
        
        for action in actions:
            color = (0, 0, 255) if action == "Sleeping" else (0, 255, 0)
            cv2.putText(image, f"- {action}", (40, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_pos += 30
        
        # Add frame number
        cv2.putText(image, f"Frame: {frame_idx}/{frame_count}", (20, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # If sleeping detected, add a visual indicator
        if "Sleeping" in actions:
            # Add a red box around the frame
            cv2.rectangle(image, (0, 0), (width-1, height-1), (0, 0, 255), 5)
            cv2.putText(image, "EYES CLOSED - SLEEPING", (width//2 - 200, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        elif eyes_closed:
            # If eyes closed but not yet counted as sleeping
            cv2.putText(image, "EYES CLOSING", (width//2 - 100, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Write frame to output video
        out.write(image)
    
    # Release resources
    cap.release()
    out.release()
    print("Video processing complete!")
    
    # Display some statistics about detected actions
    if all_actions:
        from collections import Counter
        # Flatten the list of lists to count all actions
        flat_actions = [action for sublist in all_actions for action in sublist]
        action_counts = Counter(flat_actions)
        print("\nDetected Actions Summary:")
        for action, count in action_counts.most_common():
            percentage = (count / len(flat_actions)) * 100
            print(f"{action}: {count} occurrences ({percentage:.1f}%)")
        
        # Count combinations of actions
        action_combinations = Counter([', '.join(sorted(actions)) for actions in all_actions])
        print("\nTop Action Combinations:")
        for combo, count in action_combinations.most_common(5):
            percentage = (count / len(all_actions)) * 100
            print(f"{combo}: {count} frames ({percentage:.1f}%)")
    
    return output_path

# Code to upload and process a video file
print("Please upload a video file for action recognition")
uploaded = files.upload()

if uploaded:
    # Process the first uploaded file
    video_path = next(iter(uploaded.keys()))
    output_path = process_video(video_path)
    
    # Allow user to download the processed video
    print(f"\nDownload the processed video with detected actions:")
    files.download(output_path)
else:
    print("No file was uploaded.")
