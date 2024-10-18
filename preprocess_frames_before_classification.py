import os
import time
import argparse
import cv2
from decord import VideoReader, cpu
from utils.annotation_reader import AnnotationReader
from utils.config import *
from utils.utils import create_dir_if_not_exist
from video_frame_segmentators.extract_original import ExtractOriginal
from video_frame_segmentators.extract_colours import ExtractColours
from video_frame_segmentators.extract_speed_movement import ExtractSpeedMovement
from video_frame_segmentators.extract_speed_movement_opencv import ExtractSpeedMovementMog2, ExtractSpeedMovementKNN
from video_frame_segmentators.extract_colours_and_then_speed_movement_hybrid import ExtractColoursAndThenSpeedMovementHybrid
from video_frame_segmentators.extractor import Extractor


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--approach',
    help='Choose one of approach to segment video frames',
    choices=[ExtractOriginal.APPROACH_NAME, ExtractColours.APPROACH_NAME, ExtractSpeedMovementMog2().APPROACH_NAME, ExtractSpeedMovementKNN().APPROACH_NAME, ExtractColoursAndThenSpeedMovementHybrid.APPROACH_NAME, ExtractSpeedMovement.APPROACH_NAME],
    default='original',
    type=str
)
arg_parser.add_argument('--compare_with_n_back_frame', help='Set n parameter to speed movement extraction algorithm', default=13, type=int)
args = arg_parser.parse_args()

def get_segmentation_processor(video_dir_path: str) -> Extractor:
    print('Following approach to segmentation was chosen:', args.approach)
    if args.approach == ExtractOriginal.APPROACH_NAME:
        return ExtractOriginal()
    elif args.approach == ExtractColours.APPROACH_NAME:
        return ExtractColours(cam_name=video_dir_path.split('/')[-1].split('_')[1])
    elif args.approach == ExtractSpeedMovementMog2.APPROACH_NAME:
        return ExtractSpeedMovementMog2()
    elif args.approach == ExtractSpeedMovementKNN.APPROACH_NAME:
        return ExtractSpeedMovementKNN()
    elif args.approach == ExtractColoursAndThenSpeedMovementHybrid.APPROACH_NAME:
        return ExtractColoursAndThenSpeedMovementHybrid(cam_name=video_dir_path.split('/')[-1].split('_')[1])
    elif args.approach == ExtractSpeedMovement.APPROACH_NAME:
        return ExtractSpeedMovement(compare_with_n_back_frame=args.compare_with_n_back_frame)
    else:
        raise Exception('Unexpected segmentation approach was set')

def create_necessary_directories_to_store_preprocessed_frames(approach_name: str):
    create_dir_if_not_exist(f'./data/annotated_images/segmented_frames/{approach_name}')
    create_dir_if_not_exist(f'./data/annotated_images/segmented_frames/{approach_name}/punch')
    create_dir_if_not_exist(f'./data/annotated_images/segmented_frames/{approach_name}/not_punch')

def process_video(dir_path: str, output_frame_size=(80, 80)):
    video_name = f'{dir_path.split('/')[-1].split('_')[2].upper()}'
    video = VideoReader(f'{dir_path}/data/{video_name}.mp4', ctx=cpu(0), width=output_frame_size[0], height=output_frame_size[1])
    annotation_reader = AnnotationReader(dir_path)
    segmentation_processor = get_segmentation_processor(dir_path)

    print(f'loaded {video_name} video with {len(video)} frames')
    create_necessary_directories_to_store_preprocessed_frames(segmentation_processor.get_approach_name())

    for i, frame in enumerate(video):
        frame = cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR)
        frame_label = 'punch' if annotation_reader.get_frame_label(i) != 'no_action' else 'not_punch'
        processed_img = segmentation_processor.process(frame)

        cv2.imwrite(f'./data/annotated_images/segmented_frames/{segmentation_processor.get_approach_name()}/{frame_label}/{video_name}_{i}.jpg', processed_img)

        if i % 1000 == 0:
            print(f'processed {i} frame')


def main():
    dirs_with_annotations = [path for path in os.listdir(BASE_DIR_WITH_ANNOTATIONS) if os.path.isdir(f'{BASE_DIR_WITH_ANNOTATIONS}/{path}')]

    i = 1
    for dir_with_annotations in dirs_with_annotations:
        print(f'process {i} video with path: {BASE_DIR_WITH_ANNOTATIONS}/{dir_with_annotations}')
        i += 1

        start = time.time()
        process_video(f'{BASE_DIR_WITH_ANNOTATIONS}/{dir_with_annotations}')
        end = time.time()

        print(f'processed video at: {end - start} s', )


if __name__ == '__main__':
    main()