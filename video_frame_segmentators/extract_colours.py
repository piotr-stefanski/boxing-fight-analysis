import cv2
from utils.config import HSV_map
from video_frame_segmentators.extractor import Extractor

def extract_red_from_img(img, cam_name):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, HSV_map[cam_name]['HSV_red_lower'], HSV_map[cam_name]['HSV_red_upper'])
    img_result = cv2.bitwise_and(img, img, mask=mask)
    return mask, img_result


def extract_blue_from_img(img, cam_name):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, HSV_map[cam_name]['HSV_blue_lower'], HSV_map[cam_name]['HSV_blue_upper'])
    img_result = cv2.bitwise_and(img, img, mask=mask)
    return mask, img_result

class ExtractColours(Extractor):
    APPROACH_NAME = 'extract_colours'

    def __init__(self, cam_name, debug=False):
        self.cam_name = cam_name
        self.debug = debug

    def process(self, img):
        blue_mask, blue = extract_blue_from_img(img, self.cam_name)
        red_mask, red = extract_red_from_img(img, self.cam_name)
        blue_and_red_mask = blue_mask + red_mask
        blue_and_red = cv2.bitwise_and(img, img, mask=blue_and_red_mask)

        if self.debug:
            cv2.imshow('blue and red', blue_and_red)
            cv2.imshow('blue and red mask', blue_and_red_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return blue_and_red



