# VRChat-AI-Experiments
Vision uses Yolo for object detection to detect and follow players, while keeping a specified distance from them.

It requires the yolov8n.pt file to be present in the working directory. This can be download from: https://docs.ultralytics.com/models/yolov8/

It also attempts to read nameplates using pytesseract, although this isn't really used for anything as it's not reliable enough.

Technically this could work for any game with humanoid players.