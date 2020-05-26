from collections import OrderedDict
from scipy.spatial import distance
import numpy as np


class ScipySpatialTracker:
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
            for i in range(0, len(detections)):
                self.addObject(new_object_locations[i])
        else:
            objectIDs = list(self.objects.keys())
            previous_object_locations = np.array(list(self.objects.values()))

            # pairwise distance between previous and current
            D = distance.cdist(previous_object_locations, new_object_locations)

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
