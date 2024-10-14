import time
import cv2
import utils.utils as utils

def draw_text_on_image(img, text, text_coordinate, bgr_color=(0, 0, 255), font_scale=5, thickness=3):
    cv2.putText(
        img,
        text,
        text_coordinate,
        cv2.FONT_HERSHEY_PLAIN, font_scale, bgr_color,
        thickness
    )

def main(video_path, draw_on_image=False) -> None:
    cap = cv2.VideoCapture(video_path)
    video_resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('video frames:', video_frames)
    print('video resolution', video_resolution)

    prev_time = 0
    processed_frames = 0

    config_path = 'data/models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weights_path = 'data/models/frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weights_path, config_path)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    for i in range(video_frames):
        success, img = cap.read()
        if not success:
            continue

        processed_frames += 1
        print(f'processed_frames {processed_frames}')

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        if draw_on_image:
            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        print('fps rate', fps)

        utils.is_clash_in_image(net, img, draw_rectangle_on_persons=draw_on_image)


        cv2.imshow('frame', img)
        cv2.waitKey(0)
        if cv2.waitKey(25) & 0xFF == ord('q'):
           break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(video_path='./data/videos/GH079681.MP4', draw_on_image=True)