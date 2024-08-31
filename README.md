# oak-detector
Object Detector for OAK Edge Platform

The project is based on [Luxonis OAK-D](https://shop.luxonis.com/products/oak-d-lite-1) edge device (smart camera with NPU) and [DepthAI SDK](https://docs-old.luxonis.com/projects/sdk/en/latest/index.html). Code was tested with [OAK-D Lite camera (2021 KS Edition)](https://www.kickstarter.com/projects/opencv/opencv-ai-kit-oak-depth-camera-4k-cv-edge-object-detection?ref=discovery&term=openCV&total_hits=13&category_id=338). Core functionality is object detector with [YOLO8-nano](https://github.com/ultralytics/ultralytics) model ported to NPU-specific format with this [converter](https://www.tools.luxonis.com/). That gives about 10-12 fps YOLO performance @640 image size.

**Motivation.** The aim is to optimize power consuming and thermal issues for the device especially for "EdgeAI" cases. Tests showed that witn constant NN activity device turns to thermal shoutdown after 20-25 min on about 25 C ambient temperature. Also with Paspberry Pi 5 as host the device consumed about 10000 mAh power bank in about 1,5 hour (with external rediator on the device).

**Solution.** NN activates just on demand, the trigger to activate NN is moves detector (OpenCV bacground subtractor) which is less computation-consuming. So the logic is that 70-90% of time device performs moves detection and switches to NN when moves reaches certain threshold (```t1```, no of frames). When NN performing and no objects were detected (during ```t2```, certain frame count) it will fall back to moves detector and so on. Total script duration controlled by ```t3``` parameter (actually it could be slightly longer - till the first switch between states of the system).
When NN detecting the object within frames succession (```t2``` controlled) all detections (frames with bboxes, labels and confs) are saved both as images and as records in log file in project directory.

**Tests.** Device thermal tolerance was rise from 20-25 min@25 C ambient to at less up to 2,5 hrs@35+ C ambient temperature. Power efficiency test will coming soon.

**Run Script.** First install the requirements from **requirements.txt** file with ```pip install -r requirements.txt``` Then check your DepthAI installation with ```import depthai as dai``` and then ```print(dai.__version__)``` To check OAK device see this [guide](https://docs.luxonis.com/hardware/platform/deploy/usb-deployment-guide/). 
You can run the script from Therminal with parameters:

```"-m", "--model", model path for inference (default='models/model_openvino_2022.1_6shave.blob', type=str)``` converted yolo model
```"-c", "--config", config path for inference (default='models/model.json', type=str)``` model config
```"-s", "--savepath", path to save frames with detections (default='detections', type=str)```
```"-t1", "--mmax", constant movement detecion (frames) to NN switch ON (default=25, type=int)``` 30 fps @default bghist, ksize
```"-t2", "--nodetmax", frames with no detections to OFF nn pipeline (default=25, type=int)``` 12 fps @yolov8 single class detection
```"-t3", "--maxduration", minimal total duration (sec) of algorithm cycle (default=100, type=int)```
```"-hst", "--bghist", memory buffer (frames) for bgr subtractor history (default=100, type=int)```
```"-k", "--ksize", kernel size to erode nonzero points after bgr subtractor (default=(10, 10), type=str)```
```"-o", "--out", dir to save output frames w/detections (default='./frames/', type=str)```

Linux CLI example: ```python3 oak-detector.py -t1 100 -t2 60 -t3 300 -k 7,7``` Script was tested with python 3.9 on Windows 10 and Ubuntu 20.04 LTS.

**TO DO**
1. Power autonomy test (w/RPi 5 as host)
2. Add parameters (detections filter by conf, by size...)
3. Functions docstrings

Credits: Marat A. Sabirov
Communication: avkhatovich@gmail.com

