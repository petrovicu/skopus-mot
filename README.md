# Multi-object trackers (SORT/Kalman vs scipy.spatial.distance)

Test of two different multi-object trackers:
* SORT/Kalman filter, with cross-line counting (with moving directions)
* Scipy spatial distance meter, with counting-by-tracking

Comparative testing done with YOLOv3/YOLOv2/TinyYOLO.

Cross-line counter (YOLOv3):
![](gifs/crossline-counter-yolov3.gif)

## Download YOLO weights and COCO dataset labels
https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
https://pjreddie.com/media/files/yolov3.weights
https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names