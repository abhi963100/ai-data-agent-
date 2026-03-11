from ursina import *
import cv2
import mediapipe as mp
import numpy as np
import json
import os

# --- INITIALIZE MEDIAPIPE (Optimized for Laptop) ---
mp_hands = mp.solutions.hands
# model_complexity=0 is much faster for laptops without dedicated GPUs
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Lower resolution (640x480) saves a lot of CPU power
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

app = Ursina()

# --- SETTINGS ---
current_color = color.azure
palette_colors = [color.red, color.green, color.blue, color.yellow, color.white]


class Voxel(Button):
    def __init__(self, position=(0, 0, 0), color=color.azure):
        super().__init__(
            parent=scene, position=position, model='cube',
            origin_y=0.5, texture='white_cube', color=color,
            highlight_color=color.lime
        )


# UI: Floating Color Palette
for i, c in enumerate(palette_colors):
    Entity(model='sphere', color=c, position=(-6, i * 1.5, 0), scale=0.7)

# Initial floor
for z in range(5):
    for x in range(5):
        Voxel(position=(x, 0, z), color=color.gray)

pointer = Entity(model='sphere', color=current_color, scale=0.2)
ghost = Entity(model='cube', color=color.rgba(255, 255, 255, 50), scale=1)


def update():
    global current_color
    success, frame = cap.read()
    if not success: return

    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark

        # Tracking Index Finger
        tx = (lm[8].x - 0.5) * 18
        ty = (0.5 - lm[8].y) * 12
        # Use hand size for Depth (Z)
        dist_z = np.sqrt((lm[0].x - lm[9].x) ** 2 + (lm[0].y - lm[9].y) ** 2)
        tz = (dist_z * 40) - 10

        pointer.position = lerp(pointer.position, Vec3(tx, ty, tz), 0.2)
        grid_pos = Vec3(round(pointer.x), round(pointer.y), round(pointer.z))
        ghost.position = grid_pos

        # Gestures
        # Pinch to Create (Distance between Thumb and Index)
        pinch = np.sqrt((lm[8].x - lm[4].x) ** 2 + (lm[8].y - lm[4].y) ** 2)
        # Fist to Delete
        is_fist = lm[8].y > lm[6].y and lm[12].y > lm[10].y

        if pinch < 0.05 and pointer.x > -5:
            if not any(v.position == grid_pos for v in scene.entities if isinstance(v, Voxel)):
                Voxel(position=grid_pos, color=current_color)
        elif is_fist:
            [destroy(v) for v in scene.entities if isinstance(v, Voxel) and v.position == grid_pos]

    camera.position = (2, 12, -18)
    camera.look_at(Vec3(2, 0, 2))


app.run()