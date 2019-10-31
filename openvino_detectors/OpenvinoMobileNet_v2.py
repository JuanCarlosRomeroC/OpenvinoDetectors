from openvino_detectors.openvino_detectors import OpenvinoDetector
import numpy as np

class OpenvinoMobilenet(OpenvinoDetector):
    def __init__(self, detection_threshold):
        super().__init__(cpu_lib="openvino_detectors/lib/libcpu_extension_sse4.so",
                         detector_xml="openvino_detectors/models/mobilenet_v2/frozen_inference_graph.xml",
                         detection_threshold=detection_threshold)
        # self.types = json.load(
        #     open("/home/sergej2/PycharmProjects/Recognizers/openvino_detectors/imagenet_labels.txt", "r"))
        # self.types = ["car", "bus", "truck", "van"]

    def detect(self, frame):
        cur, _, _ = self.get_detections(frame)
        # print(cur)
        # detections = cur['net_output'][0]
        # detections = cur['net_output/Softmax'][0]
        # ind = np.argmax(detections)

        return ""