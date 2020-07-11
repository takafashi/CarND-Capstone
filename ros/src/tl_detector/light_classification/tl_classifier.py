import rospy
from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf
import cv2


SSD_GRAPH_FILE = 'light_classification/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'

class TLClassifier(object):
    def __init__(self):
        self.graph = self.load_graph(SSD_GRAPH_FILE)

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.sess = tf.Session(graph=self.graph)

    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].
        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords

    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_np = np.expand_dims(image, 0)

        # Actual detection.
        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                            feed_dict={self.image_tensor: image_np})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        score_max = np.max(scores)

        confidence_cutoff = 0.8
        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)

        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        width, height = image.size
        box_coords = self.to_image_coords(boxes, height, width)

        # Each class with be represented by a differently colored box
        #draw_boxes(image, box_coords, classes)

        print("classes", classes, "scores", scores, "scores max", score_max)

        state = TrafficLight.UNKNOWN

        if score_max > confidence_cutoff:
            class_idx = classes[np.argmax(scores)]
            if class_idx == 10:
                amax = np.amax(image, axis=1)
                max_ch = np.argmax(amax)
                if max_ch == 0:
                    state = TrafficLight.RED
                elif max_ch == 1:
                    state = TrafficLight.GREEN
                else:
                    state = TrafficLight.YELLOW

            #if class_idx == 1:
            #    state = TrafficLight.GREEN
            #elif class_idx == 2:
            #    state = TrafficLight.YELLOW
            #else:
            #    state = TrafficLight.RED

        #image.save('output_model/out.jpg', quality=100)
        #time.sleep(1.8)
        if state == TrafficLight.RED:
            rospy.loginfo("TLClassifier RED")
        elif state == TrafficLight.YELLOW:
            rospy.loginfo("TLClassifier YELLOW")
        else:
            rospy.loginfo("TLClassifier GREEN")

        return state