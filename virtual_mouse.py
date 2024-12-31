import cv2
import mediapipe as mp
import pyautogui

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Capture Video from Webcam
cap = cv2.VideoCapture(0)

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

            # Get index finger tip coordinates
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Map coordinates to screen size
            screen_width, screen_height = pyautogui.size()
            screen_x = int(index_finger_tip.x * screen_width)
            screen_y = int(index_finger_tip.y * screen_height)

            # Move mouse
            pyautogui.moveTo(screen_x, screen_y)

            # Calculate distance between thumb and index finger
            distance = ((thumb_tip.x - index_finger_tip.x)**2 + (thumb_tip.y - index_finger_tip.y)**2)**0.5

            # Click if distance is below a threshold
            if distance < 0.02:
                pyautogui.click()

    # Show the frame
    cv2.imshow("Virtual Mouse", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
