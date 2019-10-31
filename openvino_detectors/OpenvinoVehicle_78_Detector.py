from openvino_detectors.OpenvinoClasffier import OpenvinoMetadata
from openvino_detectors.OpenvinoResnet import OpenvinoResnet
from openvino_detectors.openvino_detectors import OpenvinoDetector
import cv2

"""
Count frame 4770
Time 524
FPS 9
"""
PERSON = 1
CAR = 0

class OpenvinoVehicle_78_Detector(OpenvinoDetector):
    def __init__(self, detection_threshold, target_label = 1):
        super().__init__(cpu_lib="openvino_detectors/lib/libcpu_extension_sse4.so",
                         detector_xml="openvino_detectors/models/person-vehicle-bike-detection-crossroad-0078/person-vehicle-bike-detection-crossroad-0078.xml",
                         # detector_xml="openvino_detectors/models/person-vehicle-bike-detection-crossroad-0078/person-vehicle-bike-detection-crossroad-0078.xml",
                         detection_threshold=detection_threshold)
        # self.classifier = OpenvinoResnet(0)
        self.target_label = target_label

    def detect(self, frame):
        height, width = frame.shape[:-1]
        cur, new_height, new_width = self.get_detections(frame)
        detections = cur[self.d_out][0][0]

        result = []
        types = []
        for _, label, confidence, left, top, right, bottom in detections:
            if label == self.target_label:
                if confidence > self.detection_threshold:
                    left, top, right, bottom = self.convert_detections((left, top, right, bottom), height, width,
                                                                       new_height, new_width)


                    result.append((left, top, right, bottom, float(confidence)))
                    # types.append(cur_auto)
        return result
