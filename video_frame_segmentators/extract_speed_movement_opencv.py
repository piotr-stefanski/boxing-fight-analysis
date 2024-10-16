import cv2
from video_frame_segmentators.extractor import Extractor


class ExtractSpeedMovementMog2(Extractor):
    APPROACH_NAME = 'background_subtraction_by_mog2'

    def __init__(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2()

    def process(self, img):
        foreground_mask = self.backSub.apply(img)
        return cv2.bitwise_and(img, img, mask=foreground_mask)


class ExtractSpeedMovementKNN(Extractor):
    APPROACH_NAME = 'background_subtraction_by_knn'

    def __init__(self):
        self.backSub = cv2.createBackgroundSubtractorKNN()

    def process(self, img):
        foreground_mask = self.backSub.apply(img)
        return cv2.bitwise_and(img, img, mask=foreground_mask)
