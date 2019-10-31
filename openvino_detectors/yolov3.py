from math import exp

from openvino_detectors.openvino_detectors import OpenvinoDetector

"""
Convert yolov3 tensorflow to openvino
https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html 
"""
class OpenvinoYolov3(OpenvinoDetector):
    def __init__(self, detection_threshold):
        super().__init__(cpu_lib="openvino_detectors/lib/libcpu_extension_sse4.so",
                         detector_xml="openvino_detectors/models/yolov3/frozen_yolo_v3.xml",
                         detection_threshold=detection_threshold)

        self.m_input_size = 416

        self.yolo_scale_13 = 13
        self.yolo_scale_26 = 26
        self.yolo_scale_52 = 52

        self.classes = 80
        self.coords = 4
        self.num = 3
        self.anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
        self.label_text_color = (255, 255, 255)
        self.label_background_color = (125, 175, 75)
        self.box_color = (255, 128, 0)
        self.box_thickness = 1
        self.LABELS = ("person", "bicycle", "car", "motorbike", "aeroplane",
                       "bus", "train", "truck", "boat", "traffic light",
                       "fire hydrant", "stop sign", "parking meter", "bench", "bird",
                       "cat", "dog", "horse", "sheep", "cow",
                       "elephant", "bear", "zebra", "giraffe", "backpack",
                       "umbrella", "handbag", "tie", "suitcase", "frisbee",
                       "skis", "snowboard", "sports ball", "kite", "baseball bat",
                       "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                       "wine glass", "cup", "fork", "knife", "spoon",
                       "bowl", "banana", "apple", "sandwich", "orange",
                       "broccoli", "carrot", "hot dog", "pizza", "donut",
                       "cake", "chair", "sofa", "pottedplant", "bed",
                       "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                       "remote", "keyboard", "cell phone", "microwave", "oven",
                       "toaster", "sink", "refrigerator", "book", "clock",
                       "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

    def EntryIndex(self, side, lcoords, lclasses, location, entry):
        n = int(location / (side * side))
        loc = location % (side * side)
        return int(n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc)

    class DetectionObject:
        xmin = 0
        ymin = 0
        xmax = 0
        ymax = 0
        class_id = 0
        confidence = 0.0

        def __init__(self, x, y, h, w, confidence, h_scale, w_scale):
            self.xmin = int((x - w / 2) * w_scale)
            self.ymin = int((y - h / 2) * h_scale)
            self.xmax = int(self.xmin + w * w_scale)
            self.ymax = int(self.ymin + h * h_scale)
            self.confidence = confidence

    def IntersectionOverUnion(self, box_1, box_2):
        width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin)
        height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin)
        area_of_overlap = 0.0
        if (width_of_overlap_area < 0.0 or height_of_overlap_area < 0.0):
            area_of_overlap = 0.0
        else:
            area_of_overlap = width_of_overlap_area * height_of_overlap_area
        box_1_area = (box_1.ymax - box_1.ymin) * (box_1.xmax - box_1.xmin)
        box_2_area = (box_2.ymax - box_2.ymin) * (box_2.xmax - box_2.xmin)
        area_of_union = box_1_area + box_2_area - area_of_overlap
        retval = 0.0
        if area_of_union <= 0.0:
            retval = 0.0
        else:
            retval = (area_of_overlap / area_of_union)
        return retval

    def ParseYOLOV3Output(self, blob, resized_im_h, resized_im_w, original_im_h, original_im_w, threshold, objects):
        out_blob_h = blob.shape[2]
        out_blob_w = blob.shape[3]

        side = out_blob_h
        anchor_offset = 0

        if len(self.anchors) == 18:  ## YoloV3
            if side == self.yolo_scale_13:
                anchor_offset = 2 * 6
            elif side == self.yolo_scale_26:
                anchor_offset = 2 * 3
            elif side == self.yolo_scale_52:
                anchor_offset = 2 * 0

        elif len(self.anchors) == 12:  ## tiny-YoloV3
            if side == self.yolo_scale_13:
                anchor_offset = 2 * 3
            elif side == self.yolo_scale_26:
                anchor_offset = 2 * 0

        else:  ## ???
            if side == self.yolo_scale_13:
                anchor_offset = 2 * 6
            elif side == self.yolo_scale_26:
                anchor_offset = 2 * 3
            elif side == self.yolo_scale_52:
                anchor_offset = 2 * 0

        side_square = side * side
        output_blob = blob.flatten()

        for i in range(side_square):
            row = int(i / side)
            col = int(i % side)
            for n in range(self.num):
                obj_index = self.EntryIndex(side, self.coords, self.classes, n * side * side + i, self.coords)
                box_index = self.EntryIndex(side, self.coords, self.classes, n * side * side + i, 0)
                scale = output_blob[obj_index]
                if (scale < threshold):
                    continue
                x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w
                y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h
                height = exp(output_blob[box_index + 3 * side_square]) * self.anchors[anchor_offset + 2 * n + 1]
                width = exp(output_blob[box_index + 2 * side_square]) * self.anchors[anchor_offset + 2 * n]
                for j in range(self.classes):
                    # if j > 0:
                    #     break
                    class_index = self.EntryIndex(side, self.coords, self.classes, n * side_square + i,
                                                  self.coords + 1 + j)
                    prob = scale * output_blob[class_index]
                    if prob < threshold:
                        continue
                    obj = self.DetectionObject(x, y, height, width, prob, (original_im_h / resized_im_h),
                                               (original_im_w / resized_im_w))
                    objects.append(obj)

        return objects

    def get_bounding_boxes(self, values, d_h, d_w, origin_h, origin_w, threshold=0.7):
        objects = []
        for output in values:
            objects = self.ParseYOLOV3Output(output, d_h, d_w, origin_h, origin_w, threshold, objects)

        objlen = len(objects)
        for i in range(objlen):
            if objects[i].confidence == 0.0:
                continue
            for j in range(i + 1, objlen):
                if self.IntersectionOverUnion(objects[i], objects[j]) >= 0.4:
                    objects[j].confidence = 0

        bboxes = []
        for obj in objects:
            if obj.confidence > 0:
                bboxes.append((obj.xmin, obj.ymin, obj.xmax, obj.ymax, float(obj.confidence)))

        return bboxes

    def detect(self, frame):
        raw_res, _, _ = self.get_detections(frame)
        height, width = frame.shape[:-1]
        return self.get_bounding_boxes(raw_res.values(), self.d_h, self.d_w, height, width)