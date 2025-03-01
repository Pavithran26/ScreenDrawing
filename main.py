import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize drawing variables
drawing = False
last_point = None
points = []
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
color_index = 0
erase = False

# Create a window for drawing
drawing_window = np.zeros((480, 640, 3), dtype=np.uint8)

# Button positions with spacing
buttons = [
    {"text": "Blue", "x": 10, "y": 10, "color": (255, 0, 0)},
    {"text": "Green", "x": 10, "y": 60, "color": (0, 255, 0)},
    {"text": "Red", "x": 10, "y": 110, "color": (0, 0, 255)},
    {"text": "Cyan", "x": 10, "y": 160, "color": (0, 255, 255)},
    {"text": "Yellow", "x": 10, "y": 210, "color": (255, 255, 0)},
    {"text": "Magenta", "x": 10, "y": 260, "color": (255, 0, 255)},
    {"text": "Erase", "x": 10, "y": 310, "color": (0, 0, 0)},
]

# Open the camera
cap = cv2.VideoCapture(0)

# Create windows with specific positions
cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
cv2.moveWindow('Hand Tracking', 100, 100)  
cv2.namedWindow('Drawing', cv2.WINDOW_NORMAL)
cv2.moveWindow('Drawing', 800, 100)  

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # To improve performance
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the tip of the index finger
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x = int(index_tip.x * image.shape[1])
            y = int(index_tip.y * image.shape[0])

            # Check if the index finger is extended (assuming it's the drawing finger)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            if abs(thumb_tip.y - thumb_mcp.y) < abs(index_tip.y - index_mcp.y):
                drawing = True
            else:
                drawing = False

            # Adjust x coordinate for correct orientation on drawing window
            adjusted_x = drawing_window.shape[1] - x

            # Check for button presses
            for button in buttons:
                adjusted_button_x = drawing_window.shape[1] - button["x"] - 100
                if (adjusted_button_x < adjusted_x < adjusted_button_x + 100) and (button["y"] - 20 < y < button["y"] + 20):
                    if button["text"] == "Erase":
                        erase = True
                        drawing_window[:] = 0
                        points = []
                        last_point = None  
                    else:
                        color_index = buttons.index(button)
                        erase = False

            if drawing and not erase:
                points.append((adjusted_x, y))
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        cv2.line(drawing_window, points[i], points[i + 1], colors[color_index], 5)
                last_point = (adjusted_x, y)
            elif erase:
                cv2.circle(drawing_window, (adjusted_x, y), 20, (0, 0, 0), -1)
                last_point = None
            else:
                last_point = None

            # Draw the hand landmarks on the original image
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2),
            )

    # Draw buttons on the drawing window
    for button in buttons:
        adjusted_button_x = drawing_window.shape[1] - button["x"] - 100
        cv2.rectangle(drawing_window,
                      (adjusted_button_x, button["y"] - 20),
                      (adjusted_button_x + 100, button["y"] + 20),
                      button["color"], -1)
        cv2.putText(drawing_window,
                    button["text"],
                    (adjusted_button_x + 10,
                     button["y"] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255 ,255), 
                    thickness=2)

    # Display the resulting images
    cv2.imshow('Hand Tracking', image)
    cv2.imshow('Drawing', drawing_window)

    # Press 'q' to quit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
hands.close()
cap.release()
cv2.destroyAllWindows()
