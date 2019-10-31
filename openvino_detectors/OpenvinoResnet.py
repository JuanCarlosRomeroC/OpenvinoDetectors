import numpy as np
from openvino_detectors.openvino_detectors import OpenvinoDetector
import json


class OpenvinoResnet(OpenvinoDetector):
    def __init__(self, detection_threshold):
        super().__init__(cpu_lib="openvino_detectors/lib/libcpu_extension_sse4.so",
                         detector_xml="openvino_detectors/models/mobilenet/mobilenetv2-int8-tf-0001.xml",
                         detection_threshold=detection_threshold)
        self.types = json.load(
            open("/home/sergej2/PycharmProjects/Recognizers/openvino_detectors/imagenet_labels.txt", "r"))
        # self.types = ["car", "bus", "truck", "van"]

    def detect(self, frame):
        cur, _, _ = self.get_detections(frame)
        # detections = cur['net_output'][0]
        detections = cur['net_output/Softmax'][0]
        ind = np.argmax(detections)

        return self.types[ind]
