import cv2
from video_frame_segmentators.extract_colours import extract_red_from_img, extract_blue_from_img
from video_frame_segmentators.extractor import Extractor

class ExtractColoursAndThenSpeedMovementHybrid(Extractor):
    APPROACH_NAME = 'hybrid_extraction'

    def __init__(self, cam_name, debug=False):
        self.cam_name = cam_name
        self.debug = debug
        self.backSub = cv2.createBackgroundSubtractorMOG2()

    def process(self, img):
        blue_mask, blue = extract_blue_from_img(img, self.cam_name)
        red_mask, red = extract_red_from_img(img, self.cam_name)
        blue_and_red_mask = blue_mask + red_mask
        blue_and_red = cv2.bitwise_and(img, img, mask=blue_and_red_mask)

        foreground_mask = self.backSub.apply(blue_and_red)
        result = cv2.bitwise_and(blue_and_red, blue_and_red, mask=foreground_mask)

        if self.debug:
            cv2.imshow('result', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return result

