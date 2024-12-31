import cv2
import mediapipe as mp
import pyautogui

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Capture Video from Webcam
cap = cv2.VideoCapture(0)

# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return ((point1.x - point2.x)**2 + (point1.y - point2.y)**2)**0.5

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    result = hands.process(rgb_frame)

    # If hand landmarks are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions for fingers and palm
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            # Calculate distances to detect the "scroll gesture"
            distances = [
                calculate_distance(thumb_tip, wrist),
                calculate_distance(index_tip, wrist),
                calculate_distance(middle_tip, wrist),
                calculate_distance(ring_tip, wrist),
                calculate_distance(pinky_tip, wrist)
            ]

            # Scroll Gesture: All fingers wrapped close to the wrist
            if all(d < 0.1 for d in distances):  # Adjust threshold as needed
                pyautogui.scroll(-50)  # Scroll down
            else:
                # Get index finger tip coordinates for cursor movement
                screen_width, screen_height = pyautogui.size()
                screen_x = int(index_tip.x * screen_width)
                screen_y = int(index_tip.y * screen_height)

                # Move the mouse
                pyautogui.moveTo(screen_x, screen_y)

                # Optional: Add a click gesture
                distance_thumb_index = calculate_distance(thumb_tip, index_tip)
                if distance_thumb_index < 0.02:  # Click threshold
                    pyautogui.click()

    # Show the frame
    cv2.imshow("Virtual Mouse with Scrolling", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
