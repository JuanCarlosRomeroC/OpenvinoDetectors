import argparse
import json
import time

import cv2

from classifiers.OpenvinoReidClassifier import OpenvinoReidentificationPlayer
from openvino_detectors.OpenvinoMobileNet_v2 import OpenvinoMobilenet
from openvino_detectors.OpenvinoPedestrianBinary import OpenvinoPedestrian
from openvino_detectors.OpenvinoPersonDetector import OpenvinoPersonDetector
from openvino_detectors.OpenvinoVehicle_02_Detector import OpenvinoVehicle_Adas_02_Detector
from openvino_detectors.OpenvinoVehicle_78_Detector import OpenvinoVehicle_78_Detector
from openvino_detectors.yolov3 import OpenvinoYolov3
from tracking_client import TrackerClient
from visualize import visualize, drawrect


def recognize(path_to_video, detector, detections_path, nth_frame = 1):
    cap = cv2.VideoCapture(path_to_video)

    count_frame = 0
    detections = []
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # new_video = cv2.VideoWriter("visualized_video/output_detections_orange.avi", fourcc=fourcc, fps=25, apiPreference=0,
    #                             frameSize=(int(1920 / 4), int(1080 / 4)))
    all_time = 0
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            break

        if count_frame % nth_frame == 0:
            before_time = time.time()
            frame_detections = detector.detect(frame)
            all_time += time.time() - before_time
            detections.append(frame_detections)
        else:
            detections.append([])

        count_frame += 1
        if count_frame % 10 == 0:
            print(count_frame)

        # if count_frame > 200:
        #     break

    print(count_frame/all_time)
    print(all_time)
    json_detections = {"data": detections}
    json.dump(json_detections, open(detections_path, "w"))



"""
Tracker's result save in utils/tracker_result.txt
Format tracker_result.txt: for every frame list of bounding_boxes (left, top, right, bottom, track_id, phantom)}
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--name_video', required=True)
    parser.add_argument('-m', '--model_mapping', default='mapping_model.h5')
    parser.add_argument('-i', '--frame_plan', default="models/mapping/2test-5min/cafe_plan2.png")
    parser.add_argument('-vis', '--visualize', required=False, action='store_true')
    parser.add_argument('-c', '--convert', required=False, action='store_true')
    args = parser.parse_args()
    name_det = "utils/maskrcnn_test.json"
    name_tracker_result = "utils/tracker_result.txt"

    # recognize(args.name_video, OpenvinoPersonDetector(0.5), name_det)
    # recognize(args.name_video, OpenvinoVehicle_78_Detector(0.5, 2), name_det, nth_frame=20)
    # recognize(args.name_video, OpenvinoMobilenet(0.5), name_det, nth_frame=1)
    tracker_config = "utils/tracker_config_4.json"
    tracking_client = TrackerClient()
    tracking_client.run_tracker_by_detections(tracker_config, name_det, name_tracker_result,
                                              None, args.name_video)

    tracking_client.run_tracker_by_detections(tracker_config, name_det, name_tracker_result,
                                              OpenvinoReidentificationPlayer("pub_1080_3"), args.name_video)

    if args.visualize:
        visualize(args.name_video, "visualized_video/test_demo_pub.avi", name_tracker_result,tracker_config )

#     if args.convert:
#         convert_image_coord_to_plan(args.model_mapping, name_tracker_result, args.frame_plan, args.name_video)

# tracking_client = TrackerClient()
# tracking_client.run_tracker_by_detections("utils/tracker_config.json", "utils/yolov3_det.txt", "utils/tracker_result.txt",
#                                           OpenvinoReidentification(0.5), '/home/sergej2/Desktop/test_recognition.mp4')

# visualize('/home/sergej2/Desktop/test_recognition.mp4', "visualized_video/output_visualize2.avi", "utils/tracker_result.txt")