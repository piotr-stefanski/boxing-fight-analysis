import cv2
import imutils
from video_frame_segmentators.extractor import Extractor


class ExtractSpeedMovement(Extractor):
    APPROACH_NAME = 'speed_movement_extraction'

    def __init__(self, compare_with_n_back_frame: int, debug=False):
        self.compare_with_n_back_frame = compare_with_n_back_frame
        self.previous_n_images = []
        self.debug = debug
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def process(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.previous_n_images.insert(0, gray)

        if len(self.previous_n_images) <= self.compare_with_n_back_frame:
            return gray

        if self.debug:
            cv2.imshow('frame', imutils.resize(gray, width=800))

        n_back_frame = self.previous_n_images[self.compare_with_n_back_frame]
        sub = gray - n_back_frame

        # Normalization
        sub[(sub > 240) | (sub <= 15)] = 0
        sub[sub > 15] = 255
        opening = cv2.morphologyEx(sub, cv2.MORPH_OPEN, self.kernel)

        result = cv2.bitwise_and(img, img, mask=opening)

        if self.debug:
            cv2.imshow('opening', imutils.resize(opening, width=900))
            cv2.imshow('result', result)
            cv2.waitKey(0)

        # Clear memory from useless frames
        if len(self.previous_n_images) > self.compare_with_n_back_frame:
            self.previous_n_images.pop(-1)

        return result

    def get_approach_name(self):
        return f'{self.APPROACH_NAME}_{self.compare_with_n_back_frame}'
