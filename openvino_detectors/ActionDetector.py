from math import exp

from openvino_detectors.openvino_detectors import OpenvinoDetector


class OpenvinoActionDetector(OpenvinoDetector):
    def __init__(self, detection_threshold):
        super().__init__(cpu_lib="openvino_detectors/lib/libcpu_extension_sse4.so",
                         # detector_xml="openvino_detectors/models/person-detection-retail-0013/FP32/person-detection-retail-0013.xml",
                         detector_xml="openvino_detectors/models/person-detection-action-recognition-0005/person-detection-action-recognition-0005.xml",
                         detection_threshold=detection_threshold)

    class NormalizedBBox:
        def __init__(self, data):
            self.xmin = data[0]
            self.ymin = data[1]
            self.xmax = data[2]
            self.ymax = data[3]

    def detect(self, frame):
        height, width = frame.shape[:-1]
        detections = self.get_detections(frame)
        all_conf = detections["mbox_main_conf/out/conv/flat/softmax/flat"][0]
        loc_boxes = detections["mbox_loc1/out/conv/flat"][0]
        prior_boxes = detections["mbox/priorbox"][0]
        num_priors = int(prior_boxes.shape[1] / 4)
        all_detections = []
        for i in range(num_priors):
            confidence = all_conf[i * 2 + 1]
            if confidence > self.detection_threshold:
                left, top, right, bottom = self.parser_ssd_format(self.NormalizedBBox(prior_boxes[0][i * 4:]),
                                                                  self.NormalizedBBox(prior_boxes[1][i * 4:]),
                                                                  self.NormalizedBBox(loc_boxes[i * 4:]), height, width)
                all_detections.append((left, top, right, bottom, float(confidence)))
        return all_detections

    def parser_ssd_format(self, prior_bbox, variances, encoded_bbox, h, w):
        prior_width = prior_bbox.xmax - prior_bbox.xmin
        prior_height = prior_bbox.ymax - prior_bbox.ymin
        prior_center_x = (prior_bbox.xmin + prior_bbox.xmax) / 2.0
        prior_center_y = (prior_bbox.ymin + prior_bbox.ymax) / 2.0
        decoded_bbox_center_x = variances.xmin * encoded_bbox.xmin * prior_width + prior_center_x

        decoded_bbox_center_y = variances.ymin * encoded_bbox.ymin * prior_height + prior_center_y
        decoded_bbox_width = (exp((variances.xmax * encoded_bbox.xmax))) * prior_width
        decoded_bbox_height = (exp(variances.ymax * encoded_bbox.ymax)) * prior_height

        data = [decoded_bbox_center_x - decoded_bbox_width / 2.0, decoded_bbox_center_y - decoded_bbox_height / 2.0,
                decoded_bbox_center_x + decoded_bbox_width / 2.0, decoded_bbox_center_y + decoded_bbox_height / 2.0]
        decoded_bbox = self.NormalizedBBox(data)

        return max(0, decoded_bbox.xmin * w), min(decoded_bbox.ymin * h, h - 1), max(0, decoded_bbox.xmax * w), min(
            decoded_bbox.ymax * h, h - 1)
