import mss
import pygetwindow as gw
import numpy as np
import cv2
import time
import pytesseract
from pythonosc.udp_client import SimpleUDPClient
from ultralytics import YOLO
import torch
import json

def load_config():
    with open("config.json", "r") as f:
        return json.load(f)
        
config = load_config()

# OSC Setup (VRChat Controls)
osc_client = SimpleUDPClient("127.0.0.1", 9000)

def send_osc_command(direction, value):
    osc_client.send_message(f"/input/{direction}", value)

def rotate_left(steps=1):
    for _ in range(steps):
        send_osc_command("LookLeft", 1)
        time.sleep(0.1)
        send_osc_command("LookLeft", 0)

def rotate_right(steps=1):
    for _ in range(steps):
        send_osc_command("LookRight", 1)
        time.sleep(0.1)
        send_osc_command("LookRight", 0)

def move_forward():
    send_osc_command("MoveForward", 1)

def stop_forward():
    send_osc_command("MoveForward", 0)

def move_backward():
    send_osc_command("MoveBackward", 1)

def stop_backward():
    send_osc_command("MoveBackward", 0)

# Load YOLO Model with CUDA support
model = YOLO("yolov8n.pt").to("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {model.device}")

def get_game_window():
    windows = gw.getWindowsWithTitle('VRChat')
    if windows:
        game_window = windows[0]
        game_window.activate()
        return game_window.left, game_window.top, game_window.width, game_window.height
    else:
        print("Game window not found!")
        return None

def capture_screen(left, top, width, height):
    with mss.mss() as sct:
        monitor = {"top": top, "left": left, "width": width, "height": height}
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

def detect_players(frame):
    resized_frame = cv2.resize(frame, (640, 640))
    results = model(resized_frame)
    players = []
    
    scale_x = frame.shape[1] / 640
    scale_y = frame.shape[0] / 640
    
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                box_height = y2 - y1
                distance = estimate_distance(box_height)
                players.append((x1, y1, x2, y2, distance))
    
    players.sort(key=lambda p: p[4])
    return players

def estimate_distance(box_height, reference_height=200, reference_distance=1.0):
    return reference_distance * (reference_height / box_height)

def read_name_tag(frame, player_box, frame_count):
    if frame_count % 10 != 0:  # Only run OCR every 10 frames
        return ""
    
    x1, y1, x2, _ = player_box
    name_tag_region = frame[max(0, y1 - 30):y1, x1:x2]
    gray = cv2.cvtColor(name_tag_region, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 7')
    return text.strip()

def track_and_rotate(frame, width, height, last_target=None, last_direction=None, no_player_time=0, frame_count=0):
    players = detect_players(frame)
    if players:
        no_player_time = 0
        
        player_box = last_target if last_target in players else players[0]
        x1, y1, x2, y2, distance = player_box
        player_center_x = (x1 + x2) // 2
        screen_center_x = width // 2
        name_tag = read_name_tag(frame, (x1, y1, x2, y2), frame_count)
        
        print(f"Tracking Player at Distance: {distance:.2f} | Name: {name_tag}")
        
        if distance > config["max_distance"]:
            move_forward()
            stop_backward()
        elif distance < config["min_distance"]:
            move_backward()
            stop_forward()
        else:
            stop_forward()
            stop_backward()
        
        dead_zone = width * config["deadzone"]
        deviation = player_center_x - screen_center_x
        
        if deviation > dead_zone and last_direction != "right":
            print("Stepping Right...")
            rotate_right(steps=1)
            last_direction = "right"
        elif deviation < -dead_zone and last_direction != "left":
            print("Stepping Left...")
            rotate_left(steps=1)
            last_direction = "left"
        else:
            last_direction = None
        
        last_target = player_box
    else:
        print("No player detected. Stopping all movement.")
        stop_forward()
        stop_backward()
        no_player_time += 1
        if no_player_time > 50:
            print("Searching for player...")
            rotate_right(steps=1)
        last_direction = None
        last_target = None
    
    return last_target, last_direction, no_player_time

def main():
    time.sleep(2)
    game_window = get_game_window()
    if not game_window:
        return
    left, top, width, height = game_window

    last_target = None
    last_direction = None
    no_player_time = 0
    frame_count = 0

    with mss.mss() as sct:
        while True:
            monitor = {"top": top, "left": left, "width": width, "height": height}
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            last_target, last_direction, no_player_time = track_and_rotate(
                frame, width, height, last_target, last_direction, no_player_time, frame_count
            )
            
            cv2.imshow("YOLO Player Detection", frame)
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()
    stop_forward()
    stop_backward()

if __name__ == "__main__":
    main()
