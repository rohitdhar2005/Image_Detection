# 🎯 Object Detection using OpenCV

A Python-based object detection system using OpenCV. This project uses pre-trained deep learning models like YOLO or MobileNet SSD for detecting objects in images and video streams.

## 🚀 Features

- 📷 Real-time object detection using webcam or video files
- 🧠 Supports YOLO and MobileNet SSD models
- 🗂️ Detects multiple common object classes (person, car, dog, etc.)
- 📏 Displays bounding boxes and class labels
- 💾 Save detection output as images or videos

## 🛠️ Tech Stack

- **Python 3.6+**
- **OpenCV** – for image and video processing
- **NumPy** – for numerical operations
- **Pre-trained models** – YOLOv3/v4 or MobileNet SSD

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/object-detection-opencv.git
   cd object-detection-opencv
   ```

2. **Install dependencies**
   ```bash
   pip install opencv-python numpy
   ```

3. **Download model files**
   - For **YOLO**:
     - `yolov4.cfg`
   ```bash
   https://github.com/AlexeyAB/darknet/raw/master/cfg/yolov4.cfg
   ```
     - `yolov4.weights`
   ```bash
   https://objects.githubusercontent.com/github-production-release-asset-2e65be/75388965/4b8a4e00-b2d7-11eb-900f-678196af5945?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20250412%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250412T071350Z&X-Amz-Expires=300&X-Amz-Signature=186a2ec99150b10c2a21a13d8c3e448682394512689391c8bba39e7ac7e5f3ea&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3Dyolov4.weights&response-content-type=application%2Foctet-stream
   ```
     - `coco.names`
   - For **MobileNet SSD**:
     - `deploy.prototxt`
     - `mobilenet_iter_73000.caffemodel`

   *(Place them in the `models/` directory)*

## ▶️ Usage

Run object detection on webcam:
```bash
python detect.py --model yolov3 --source webcam
```

Run on a video file:
```bash
python detect.py --model mobilenet --source path/to/video.mp4
```

Run on an image:
```bash
python detect.py --model yolov3 --source path/to/image.jpg
```

## 📸 Sample Output

> ![image](https://github.com/user-attachments/assets/fdb1bbb2-f7e4-4f7e-bc4e-3b9b45512284)


## ⚙️ Command-Line Arguments

- `--model` : Model to use (`yolov3` or `mobilenet`)
- `--source` : Source of input (`webcam`, image path, or video path)
- `--save` : Optional flag to save output to disk

## 💡 Future Enhancements

- Integration with live CCTV streams
- GUI interface for detection and visualization
- Object tracking (Deep SORT)
- Custom object detection model support (YOLOv8, etc.)

## 🤝 Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add NewFeature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Built with 🔍 and ❤️ using OpenCV and Python.

