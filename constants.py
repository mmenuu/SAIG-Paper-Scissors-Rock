import cv2

x, y, w, h = 334, 71, 121, 121

model_path = "model/modelTrue/model2/model2.json"
model_weights_path = "model/modelTrue/model2/best_weights.h5"

BG_path = "src/BG2.png"
FRM_path = "src/frame.png"

rectangle_color = (153, 76, 0)
text_color = (153, 76, 0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

computer_gestures = {
    "rock":     "src/1.png",
    "paper":    "src/2.png",
    "scissors": "src/3.png"
}

stronger_gesture = {
    "rock":  "paper",
    "paper": "scissor",
    "scissors": "rock"
}