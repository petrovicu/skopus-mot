import cv2 as cv
import numpy as np
import time
from scipy_distance_tracker import ScipySpatialTracker


def draw_bounding_boxes(idxs, bboxes, classIDs, confidences, detections_bboxes):
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (bboxes[i][0], bboxes[i][1])
            (w, h) = (bboxes[i][2], bboxes[i][3])
            detections_bboxes.append((x, y, x + w, y + h))
            clr = [int(c) for c in bbox_colors[classIDs[i]]]
            cv.rectangle(image, (x, y), (x + w, y + h), clr, 2)
            cv.putText(image, "{}: {:.4f}".format(labels[classIDs[i]], confidences[i]),
                       (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)


def draw_id_label(objects, class_id, tracker_obj):
    for (object_id, centroid) in objects.items():
        text_label = "ID {}".format(object_id)
        tracker_obj.number_of_elements = max(tracker_obj.number_of_elements, object_id)
        clr = [int(c) for c in bbox_colors[class_id]]
        cv.putText(image, text_label, (centroid[0] - 10, centroid[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
        cv.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


model_type = "yolov3"

if model_type == "yolov2":
    yolomodel = {"config_path": "E:\models\yolov2\yolov2.cfg",
                 "model_weights_path": "E:\models\yolov2\yolov2.weights",
                 "coco_names": "E:\models\yolov2\coco.names",
                 "confidence_threshold": 0.5,
                 "threshold": 0.3
                 }
if model_type == "yolov2-tiny":
    yolomodel = {"config_path": "E:\models\yolo-tiny\yolov2-tiny.cfg",
                 "model_weights_path": "E:\models\yolo-tiny\yolov2-tiny.weights",
                 "coco_names": "E:\models\yolo-tiny\coco.names",
                 "confidence_threshold": 0.5,
                 "threshold": 0.3
                 }
if model_type == "yolov3":
    yolomodel = {"config_path": "E:\models\yolov3\yolov3.cfg",
                 "model_weights_path": "E:\models\yolov3\yolov3.weights",
                 "coco_names": "E:\models\yolov3\coco.names",
                 "confidence_threshold": 0.5,
                 "threshold": 0.3
                 }

number_of_pedestrians = 0
number_of_cars = 0

net = cv.dnn.readNetFromDarknet(yolomodel["config_path"], yolomodel["model_weights_path"])
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

labels = open(yolomodel["coco_names"]).read().strip().split("\n")

np.random.seed(12345)
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print(layer_names)

bbox_colors = np.random.randint(0, 255, size=(len(labels), 3))

maxLost = 5  # maximum number of object losts counted when the object is being tracked
pedestrian_tracker = ScipySpatialTracker(maxLost=maxLost)
car_tracker = ScipySpatialTracker(maxLost=maxLost)

video_src = "input/crossroads.mp4"
cap = cv.VideoCapture(video_src)

(H, W) = (None, None)  # input image height and width for the network
writer = None

loop_counter = 0

while True:
    loop_counter += 1
    start_time = time.time()
    ok, image = cap.read()

    if not ok:
        print("Cannot read the video feed.")
        break

    if W is None or H is None:
        (H, W) = image.shape[:2]

    blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections_layer = net.forward(layer_names)  # detect objects using object detection model

    pedestrian_detections_bbox = []  # pedestrian bounding box for detections
    car_detections_bbox = []  # car bounding box for detections

    pedestrian_boxes, pedestrian_confidences, pedestrian_classIDs = [], [], []
    car_boxes, car_confidences, car_classIDs = [], [], []

    for out in detections_layer:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # we are not interested in other COCO classes
            if classID > 3:
                continue

            if confidence > yolomodel['confidence_threshold']:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                if classID == 0:  # pedestrians
                    pedestrian_boxes.append([x, y, int(width), int(height)])
                    pedestrian_confidences.append(float(confidence))
                    pedestrian_classIDs.append(classID)
                if classID == 2:  # cars
                    car_boxes.append([x, y, int(width), int(height)])
                    car_confidences.append(float(confidence))
                    car_classIDs.append(classID)

    pedestrians_idxs = cv.dnn.NMSBoxes(pedestrian_boxes, pedestrian_confidences, yolomodel["confidence_threshold"],
                                       yolomodel["threshold"])
    cars_idxs = cv.dnn.NMSBoxes(car_boxes, car_confidences, yolomodel["confidence_threshold"], yolomodel["threshold"])

    draw_bounding_boxes(pedestrians_idxs, pedestrian_boxes, pedestrian_classIDs, pedestrian_confidences,
                        pedestrian_detections_bbox)
    draw_bounding_boxes(cars_idxs, car_boxes, car_classIDs, car_confidences, car_detections_bbox)

    pedestrian_objects = pedestrian_tracker.update(pedestrian_detections_bbox)  # update with newly detected objects
    car_objects = car_tracker.update(car_detections_bbox)  # update tracker with newly detected objects

    draw_id_label(pedestrian_objects, 0, pedestrian_tracker)
    draw_id_label(car_objects, 2, car_tracker)

    pedestrian_clr = [int(c) for c in bbox_colors[0]]
    car_clr = [int(c) for c in bbox_colors[2]]
    cv.putText(image, "Peds: " + str(pedestrian_tracker.number_of_elements + 1), (30, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, pedestrian_clr, 3)
    cv.putText(image, "Cars: " + str(car_tracker.number_of_elements + 1), (30, 70),
               cv.FONT_HERSHEY_SIMPLEX, 1, car_clr, 3)
    cv.imshow("image", image)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    if writer is None:
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        writer = cv.VideoWriter("E:/Skopus/recordings/output/yolo/kg_nms_" + model_type + ".avi", fourcc, 20,
                                (W, H), True)
    writer.write(image)
    print("FPS: ", 1.0 / (time.time() - start_time))

writer.release()
cap.release()
cv.destroyWindow("image")
