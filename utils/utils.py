import math
from utils.boxer_detector import extract_boxers_from_persons, extract_red_boxers_from_persons, extract_blue_boxers_from_persons, extract_boxers_on_one_box_from_persons
import cv2
from pathlib import Path


def person_is_in_the_ring(person_box, image_height):
    down_line_y_val = person_box[1] + person_box[3]
    threshold = image_height/5*3

    return down_line_y_val > threshold


def are_persons_close_together(person_boxs, image_width, distance_to_width_ratio=6):
    distances = []

    for i, person_box in enumerate(person_boxs):
        if i == len(person_boxs) - 1:
            distances.append(get_distance_between_two_center_of_boxes(person_box, person_boxs[0]))
        else:
            distances.append(get_distance_between_two_center_of_boxes(person_box, person_boxs[i + 1]))

    result = any([distance < (image_width / distance_to_width_ratio) for distance in distances])

    return result


def get_distance_between_two_center_of_boxes(box1: list, box2: list):
    center_box1 = get_center_of_box(box1)
    center_box2 = get_center_of_box(box2)

    return math.dist(center_box1, center_box2)


def get_center_of_box(box: list):
    x = box[0] + box[2] / 2
    y = box[1] + box[3] / 2
    return [x, y]


def get_boxes_only_with_persons_from_ring(zipped_results, image_height):
    return [
        (classId, confidence, box)
        for classId, confidence, box
        in zipped_results
        if classId == 1 and person_is_in_the_ring(box, image_height)
    ]


def get_boxes_only_with_persons(zipped_results):
    return [
        (classId, confidence, box)
        for classId, confidence, box
        in zipped_results
        if classId == 1
    ]


def is_clash_in_image(net, img, threshold=0.60, draw_rectangle_on_persons=False) -> {}:
    height, width, channels = img.shape

    # step_1
    classIds, confs, bbox = net.detect(img, confThreshold=threshold)
    if len(classIds) == 0:
        return 0, 0, 0, False, 0

    # step_2
    zipped_results = get_boxes_only_with_persons_from_ring(zip(classIds.flatten(), confs.flatten(), bbox), height)
    count_of_all_persons = len(get_boxes_only_with_persons(zip(classIds.flatten(), confs.flatten(), bbox)))
    count_persons_on_ring = len([
        (classId, confidence, box)
        for classId, confidence, box
        in zipped_results
        if person_is_in_the_ring(box, height)
    ])
    persons_on_ring = get_coordinates_from_zipped_results(zipped_results)

    # step_3
    red_boxers_on_ring = extract_red_boxers_from_persons(persons_on_ring, img)
    blue_boxers_on_ring = extract_blue_boxers_from_persons(persons_on_ring, img)
    boxers_on_ring = red_boxers_on_ring + blue_boxers_on_ring

    boxers_on_one_box = extract_boxers_on_one_box_from_persons(persons_on_ring, img)

    if draw_rectangle_on_persons:
        # threshold line
        img = cv2.line(img, (0, int(height / 5 * 3)),
                       (width, int(height / 5 * 3)),
                       (0, 0, 255), 5)

        for box in red_boxers_on_ring:
            cv2.rectangle(img, box, color=(0, 0, 255), thickness=2)

        for box in blue_boxers_on_ring:
            cv2.rectangle(img, box, color=(255, 0, 0), thickness=2)

    len_boxers_on_ring = len(boxers_on_ring) if len(boxers_on_ring) <= 2 else 2

    # is_clash = len(red_boxers_on_ring) >= 1 and len(blue_boxers_on_ring) >= 1 \
    #   and are_persons_close_together([red_boxers_on_ring[0], blue_boxers_on_ring[0]], width)

    is_clash = ((len(red_boxers_on_ring) >= 1 and len(blue_boxers_on_ring) >= 1) or boxers_on_one_box == 1) \
        and are_persons_close_together([red_boxers_on_ring[0], blue_boxers_on_ring[0]], width)

    distance_between_boxers = get_distance_between_two_center_of_boxes(red_boxers_on_ring[0], blue_boxers_on_ring[0]) \
        if len(red_boxers_on_ring) >= 1 and len(blue_boxers_on_ring) >= 1 else None

    # print(
    #     f'all persons: {count_of_all_persons} ;; persons on ring: {count_persons_on_ring} ;; boxers on ring: {len_boxers_on_ring}')
    return count_of_all_persons, count_persons_on_ring, len_boxers_on_ring, is_clash, distance_between_boxers


def get_coordinates_from_zipped_results(zipped_results):
    # third index contains coordinates, first detected class (in this case always person(1)), second detection precision
    return [el[2] for el in zipped_results]


def create_dir_if_not_exist(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
