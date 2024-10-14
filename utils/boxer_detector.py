import cv2
import numpy as np

HSV_red_lower = np.array([0, 179, 58])
HSV_red_upper = np.array([16, 255, 255])
HSV_blue_lower = np.array([115, 0, 0])
HSV_blue_upper = np.array([147, 255, 72])


def extract_boxers_from_persons(persons: list, img) -> list:
    return [
        person for person in persons
        if boxer_is_in_box(
            cut_person_from_image(person, img)
        )]


def extract_red_boxers_from_persons(persons: list, img) -> list:
    return [
        person for person in persons
        if red_boxer_is_in_box(
            cut_person_from_image(person, img)
        )]


def extract_blue_boxers_from_persons(persons: list, img) -> list:
    return [
        person for person in persons
        if blue_boxer_is_in_box(
            cut_person_from_image(person, img)
        )]


def extract_boxers_on_one_box_from_persons(persons: list, img) -> list:
    return [
        person for person in persons
        if blue_and_red_boxers_are_in_box(
            cut_person_from_image(person, img)
        )]


def cut_person_from_image(person: list, img):
    # 0 -> xmin, 1 -> ymin, 2 -> box width, 3 -> box height
    return img[person[1]:person[1] + person[3], person[0]:person[0] + person[2]]


def boxer_is_in_box(img) -> bool:
    img_height = img.shape[0]
    img = img[0:0 + int(img_height / 5), :]

    return \
        is_image_contains_color_over_threshold(img, HSV_red_lower, HSV_red_upper, 'red') \
        or is_image_contains_color_over_threshold(img, HSV_blue_lower, HSV_blue_upper, 'blue')


def red_boxer_is_in_box(img) -> bool:
    img_height = img.shape[0]
    img = img[0:0 + int(img_height / 5), :]

    return is_image_contains_color_over_threshold(img, HSV_red_lower, HSV_red_upper, 'red')


def blue_boxer_is_in_box(img) -> bool:
    img_height = img.shape[0]
    img = img[0:0 + int(img_height / 5), :]

    return is_image_contains_color_over_threshold(img, HSV_blue_lower, HSV_blue_upper, 'blue')


def blue_and_red_boxers_are_in_box(img) -> bool:
    img_height = img.shape[0]
    img = img[0:0 + int(img_height / 5), :]

    return \
        is_image_contains_color_over_threshold(img, HSV_red_lower, HSV_red_upper, 'red') \
        and is_image_contains_color_over_threshold(img, HSV_blue_lower, HSV_blue_upper, 'blue')


def is_image_contains_color_over_threshold(img, lower_hsv, upper_hsv, wanted_color, threshold=0.1):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

    count_white_pixels_on_mask = np.count_nonzero(mask)
    count_black_pixels_on_mask = mask.size - count_white_pixels_on_mask

    return count_white_pixels_on_mask / count_black_pixels_on_mask > threshold

