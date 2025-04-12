import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk, Text, scrolledtext
from PIL import Image, ImageTk
import os

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Object Detection")
        self.root.geometry("1000x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Variables
        self.running = False
        self.camera_active = False
        self.cap = None
        self.detection_thread = None
        self.confidence_threshold = 0.5
        
        # Create coco.names file if it doesn't exist first
        self.create_coco_names_if_needed()
        
        # Load COCO class names
        self.classes = []
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Load YOLOv4 network
        self.config_path = "yolov4.cfg"
        self.weights_path = "yolov4.weights"
        
        # Set up UI components
        self.setup_ui()
        
        # Status messages
        self.log_message("Application started")
        self.log_message("Please download YOLOv4 files if not already present:")
        self.log_message("- yolov4.cfg: https://github.com/AlexeyAB/darknet/raw/master/cfg/yolov4.cfg")
        self.log_message("- yolov4.weights: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.weights")
        self.log_message("Click 'Start Camera' to begin detection")
        
    def create_coco_names_if_needed(self):
        """Create a default coco.names file if it doesn't exist"""
        if not os.path.exists("coco.names"):
            self.create_default_coco_names()
            
    def create_default_coco_names(self):
        """Create a default coco.names file"""
        with open("coco.names", "w") as f:
            f.write("""person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush""")
    
    def setup_ui(self):
        """Set up the UI components"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Camera controls
        ttk.Button(controls_frame, text="Start Camera", command=self.start_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Stop Camera", command=self.stop_camera).pack(side=tk.LEFT, padx=5)
        
        # Confidence threshold
        ttk.Label(controls_frame, text="Confidence Threshold:").pack(side=tk.LEFT, padx=(20, 5))
        self.threshold_var = tk.DoubleVar(value=0.5)
        threshold_scale = ttk.Scale(controls_frame, from_=0.1, to=1.0, length=200, 
                                   variable=self.threshold_var, orient=tk.HORIZONTAL,
                                   command=self.update_threshold)
        threshold_scale.pack(side=tk.LEFT, padx=5)
        self.threshold_label = ttk.Label(controls_frame, text="0.50")
        self.threshold_label.pack(side=tk.LEFT, padx=5)
        
        # Middle section - split into video and detection list
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Video frame
        video_frame = ttk.LabelFrame(middle_frame, text="Camera Feed")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Detection list frame - Fixed: removed width parameter from pack()
        detections_frame = ttk.LabelFrame(middle_frame, text="Detected Objects")
        detections_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        
        # Set width of detections_frame using a dummy frame
        dummy_frame = ttk.Frame(detections_frame, width=300)
        dummy_frame.pack(side=tk.TOP)
        
        self.detection_text = scrolledtext.ScrolledText(detections_frame, wrap=tk.WORD, width=30, height=20)
        self.detection_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Log frame
        log_frame = ttk.LabelFrame(main_frame, text="Log")
        log_frame.pack(fill=tk.X, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=5)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(5, 0))
    
    def update_threshold(self, value):
        """Update the confidence threshold value"""
        self.confidence_threshold = float(value)
        self.threshold_label.config(text=f"{self.confidence_threshold:.2f}")
    
    def log_message(self, message):
        """Add a message to the log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        
    def update_status(self, message):
        """Update the status bar"""
        self.status_var.set(message)
        
    def update_detection_list(self, detections):
        """Update the detection list"""
        self.detection_text.delete(1.0, tk.END)
        if not detections:
            self.detection_text.insert(tk.END, "No objects detected")
            return
            
        # Sort by confidence
        detections_sorted = sorted(detections, key=lambda x: x[1], reverse=True)
        
        for i, (label, confidence) in enumerate(detections_sorted):
            self.detection_text.insert(tk.END, f"{i+1}. {label} ({confidence:.2f})\n")
    
    def start_camera(self):
        """Start the camera and detection"""
        if self.camera_active:
            self.log_message("Camera is already running")
            return
            
        try:
            # Check if YOLO files exist
            if not os.path.exists(self.config_path) or not os.path.exists(self.weights_path):
                self.log_message("YOLO model files not found. Please download them first.")
                self.update_status("Error: Model files missing")
                return
            
            # Try to load the YOLO model
            self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
            
            # Use CPU backend only - avoid CUDA that's causing errors
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.log_message("Using CPU backend for compatibility")
                
            # Get output layer names
            layer_names = self.net.getLayerNames()
            try:
                # OpenCV 4.5.4+
                self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            except:
                # Older OpenCV versions
                self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
                
            # Start camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
                
            self.running = True
            self.camera_active = True
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            self.update_status("Camera active - detecting objects")
            self.log_message("Camera started successfully")
            
        except Exception as e:
            self.log_message(f"Error starting camera: {str(e)}")
            if "readNetFromDarknet" in str(e):
                self.log_message("YOLO model files not found. Please download them first.")
            self.update_status("Error: Could not start camera")
    
    def stop_camera(self):
        """Stop the camera and detection"""
        if not self.camera_active:
            self.log_message("Camera is not running")
            return
            
        self.running = False
        self.camera_active = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        # Clear the video display
        self.video_label.config(image='')
        
        self.update_status("Camera stopped")
        self.log_message("Camera stopped")
        
    def on_close(self):
        """Handle window close event"""
        self.stop_camera()
        self.root.destroy()
        
    def detection_loop(self):
        """Main detection loop"""
        while self.running:
            try:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    self.log_message("Failed to capture frame")
                    break
                    
                # Create blob from image
                height, width, _ = frame.shape
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
                
                # Set input and forward pass
                self.net.setInput(blob)
                outputs = self.net.forward(self.output_layers)
                
                # Process outputs
                class_ids = []
                confidences = []
                boxes = []
                
                for output in outputs:
                    for detection in output:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        
                        if confidence > self.confidence_threshold:
                            # Scale bounding box coordinates back to original image size
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            
                            # Get top-left corner coordinates of bounding box
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                
                # Apply non-maximum suppression to remove redundant boxes
                indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
                
                # Process and display detections
                detections = []
                
                if len(indices) > 0:
                    # Handle different versions of OpenCV
                    try:
                        indices = indices.flatten()
                    except:
                        pass
                        
                    for i in indices:
                        box = boxes[i]
                        x, y, w, h = box
                        
                        # Ensure we don't go out of bounds
                        x = max(0, x)
                        y = max(0, y)
                        
                        # Draw bounding box
                        color = (0, 255, 0)  # Green
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        
                        # Put label and confidence
                        label = self.classes[class_ids[i]]
                        confidence = confidences[i]
                        text = f"{label}: {confidence:.2f}"
                        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Add to detections list
                        detections.append((label, confidence))
                
                # Update UI with detected objects
                self.root.after(0, self.update_detection_list, detections)
                
                # Convert to RGB for tkinter
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PhotoImage
                img = Image.fromarray(rgb_frame)
                img = img.resize((640, 480), Image.LANCZOS)  # Resize for display
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Update video display
                self.video_label.img_tk = img_tk  # Keep a reference to prevent garbage collection
                self.video_label.config(image=img_tk)
                
            except Exception as e:
                self.log_message(f"Error in detection loop: {str(e)}")
                break
        
        self.log_message("Detection loop ended")

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
