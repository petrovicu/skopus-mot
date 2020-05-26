import cv2 as cv
from scipy.spatial import distance
import numpy as np
from collections import OrderedDict
import time


class Tracker:
    def __init__(self, maxLost=30):  # maxLost: maximum object lost counted when the object is being tracked
        self.nextObjectID = 0  # ID of next object
        self.objects = OrderedDict()  # stores ID:Locations
        self.lost = OrderedDict()  # stores ID:Lost_count

        self.number_of_elements = 0

        self.maxLost = maxLost  # maximum number of frames object was not detected.

    def addObject(self, new_object_location):
        self.objects[self.nextObjectID] = new_object_location  # store new object location
        self.lost[self.nextObjectID] = 0  # initialize frame_counts for when new object is undetected

        self.nextObjectID += 1

    def removeObject(self, objectID):  # remove tracker data after object is lost
        del self.objects[objectID]
        del self.lost[objectID]

    @staticmethod
    def getLocation(bounding_box):
        xlt, ylt, xrb, yrb = bounding_box
        return (int((xlt + xrb) / 2.0), int((ylt + yrb) / 2.0))

    def update(self, detections):
        if len(detections) == 0:  # if no object detected in the frame
            lost_ids = list(self.lost.keys())
            for objectID in lost_ids:
                self.lost[objectID] += 1
                if self.lost[objectID] > self.maxLost:
                    self.removeObject(objectID)

            return self.objects

        new_object_locations = np.zeros((len(detections), 2), dtype="int")  # current object locations

        for (i, detection) in enumerate(detections):
            new_object_locations[i] = self.getLocation(detection)

        if len(self.objects) == 0:
            for i in range(0, len(detections)): self.addObject(new_object_locations[i])
        else:
            objectIDs = list(self.objects.keys())
            previous_object_locations = np.array(list(self.objects.values()))

            D = distance.cdist(previous_object_locations,
                               new_object_locations)  # pairwise distance between previous and current

            row_idx = D.min(axis=1).argsort()  # (minimum distance of previous from current).sort_as_per_index

            cols_idx = D.argmin(axis=1)[row_idx]  # index of minimum distance of previous from current

            assignedRows, assignedCols = set(), set()

            for (row, col) in zip(row_idx, cols_idx):

                if row in assignedRows or col in assignedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = new_object_locations[col]
                self.lost[objectID] = 0

                assignedRows.add(row)
                assignedCols.add(col)

            unassignedRows = set(range(0, D.shape[0])).difference(assignedRows)
            unassignedCols = set(range(0, D.shape[1])).difference(assignedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unassignedRows:
                    objectID = objectIDs[row]
                    self.lost[objectID] += 1

                    if self.lost[objectID] > self.maxLost:
                        self.removeObject(objectID)

            else:
                for col in unassignedCols:
                    self.addObject(new_object_locations[col])

        return self.objects


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

maxLost = 5  # maximum number of object lost counted when the object is being tracked
pedestrian_tracker = Tracker(maxLost=maxLost)
car_tracker = Tracker(maxLost=maxLost)

video_src = "input/crossroads.mp4"
cap = cv.VideoCapture(video_src)

(H, W) = (None, None)  # input image height and width for the network
writer = None

loop_counter = 0

while True:
    loop_counter += 1
    start_time = time.time()
    ok, image = cap.read()

    # crop image
    # image = image[200:600, 100:475]

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

            if classID > 3:
                continue

            confidence = scores[classID]

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
    print("Loop counter: ", loop_counter)

writer.release()
cap.release()
cv.destroyWindow("image")
