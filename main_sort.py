# import the necessary packages
import argparse
import os
import time

import cv2
import imutils

from sort_kalman_tracker import *

tracker = Sort()
memory = {}

'''
# PEOPLE.MP4
line = [(0, 450), (1920, 450)]
'''

# CROSSROADS.MP4
car_line_1 = [(180, 376), (312, 377)]
car_line_2 = [(424, 372), (497, 399)]
car_line_3 = [(764, 496), (800, 444)]
car_line_4 = [(178, 429), (298, 527)]

towards_counter = 0
away_from_counter = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["E:/models/yolov3/", "yolov3.weights"])
configPath = os.path.sep.join(["E:/models/yolov3/", "yolov3.cfg"])
labelsPath = os.path.sep.join(["E:/models/yolov3/", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
video_src = "input/crossroads.mp4"
vs = cv2.VideoCapture(video_src)
writer = None
(W, H) = (None, None)

frameIndex = 0

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)

            # skip post-processing other COCO classes
            if classID > 3:
                continue

            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    dets = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x + w, y + h, confidences[i]])

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    tracks = tracker.update(dets)

    boxes = []
    indexIDs = []
    c = []
    previous = memory.copy()
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    if len(boxes) > 0:
        i = int(0)
        for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            if indexIDs[i] in previous:
                previous_box = previous[indexIDs[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                p_current = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                p_previous = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                cv2.line(frame, p_current, p_previous, color, 3)

                if intersect(p_current, p_previous, car_line_1[0], car_line_1[1]):
                    towards_counter += 1
                if intersect(p_current, p_previous, car_line_2[0], car_line_2[1]):
                    towards_counter += 1
                if intersect(p_current, p_previous, car_line_3[0], car_line_3[1]):
                    towards_counter += 1
                if intersect(p_current, p_previous, car_line_4[0], car_line_4[1]):
                    towards_counter += 1
                    # if p_current[1] > p_previous[1]:
                    #     towards_counter += 1
                    # else:
                    #     away_from_counter += 1
            # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            text = "{}".format(indexIDs[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            i += 1

    # draw line
    cv2.line(frame, car_line_1[0], car_line_1[1], (0, 255, 255), 3)
    cv2.line(frame, car_line_2[0], car_line_2[1], (0, 255, 255), 3)
    cv2.line(frame, car_line_3[0], car_line_3[1], (0, 255, 255), 3)
    cv2.line(frame, car_line_4[0], car_line_4[1], (0, 255, 255), 3)

    # draw counter
    '''
    # PEOPLE.MP4
    cv2.putText(frame, "To: " + str(towards_counter), (50, 100), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 255, 255), 5)
    cv2.putText(frame, "From:" + str(away_from_counter), (50, 200), cv2.FONT_HERSHEY_DUPLEX, 3.0, (255, 0, 255), 5)
    '''

    # CROSSROADS.MP4
    cv2.putText(frame, "Cars: " + str(towards_counter), (50, 100), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 0, 255), 3)

    # counter += 1

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("E:/Skopus/recordings/output/skopus-mot-with-yolo/crossroads-sort-yolov3.avi",
                                 fourcc, 20, (frame.shape[1], frame.shape[0]), True)

        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * total))

    # write the output frame to disk
    writer.write(frame)

    # increase frame index
    frameIndex += 1

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
