# oak-detector
Object Detector for OAK Edge Platform

The project is based on [Luxonis OAK-D](https://shop.luxonis.com/products/oak-d-lite-1) edge device (smart camera with NPU) and [DepthAI SDK](https://docs-old.luxonis.com/projects/sdk/en/latest/index.html). Code was tested with [OAK-D Lite camera (2021 KS Edition)](https://www.kickstarter.com/projects/opencv/opencv-ai-kit-oak-depth-camera-4k-cv-edge-object-detection?ref=discovery&term=openCV&total_hits=13&category_id=338). Core functionality is object detector with [YOLO8-nano](https://github.com/ultralytics/ultralytics) model ported to NPU-specific format with this [converter](https://www.tools.luxonis.com/). That gives about 10-12 fps YOLO performance @640 image size.

**Motivation.** The aim is to optimize power consuming and thermal issues for the device especially for "EdgeAI" cases. Tests showed that witn constant NN activity device turns to thermal shoutdown after 20-25 min on about 25 C ambient temperature. Also with Paspberry Pi 5 as host the device consumed about 10000 mAh power bank in about 1,5 hour (with external rediator on the device).

**Solution.** NN activates just on demand, the trigger to activate NN is moves detector (OpenCV bacground subtractor) which is less computation-consuming. So the logic is that 70-90% of time device performs moves detection and switches to NN when moves reaches certain threshold (```t1```, no of frames). When NN performing and no objects were detected (during ```t2```, certain frame count) it will fall back to moves detector and so on. Total script duration controlled by ```t3``` parameter (actually it could be slightly longer - till the first switch between states of the system).
When NN detecting the object within frames succession (```t2``` controlled) all detections (frames with bboxes, labels and confs) are saved both as images and as records in log file in project directory.

**Tests.** Device thermal tolerance was rise from 20-25 min@25 C ambient to at less up to 2,5 hrs@35+ C ambient temperature. Power efficiency test will coming soon.

**Run Script.** First install the requirements from **requirements.txt** file with ```pip install -r requirements.txt``` Then check your OAK device by importing DepthAI and command   You can run the script from Therminal with parameters:
Script was tested with python 3.9 on Windows 10 and Ubuntu 20.04 LTS.

