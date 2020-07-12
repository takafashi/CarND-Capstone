from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf


SSD_GRAPH_FILE = 'light_classification/frozen_model/frozen_inference_graph.pb'
CONFIDENCE_CUTOFF = 0.8

class TLClassifier(object):
    def __init__(self):
        self.graph = self.load_graph(SSD_GRAPH_FILE)

        # The input placeholder for the image.
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
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

        # Filter boxes with a confidence score less than `CONFIDENCE_CUTOFF`
        boxes, scores, classes = self.filter_boxes(CONFIDENCE_CUTOFF, boxes, scores, classes)

        state = TrafficLight.UNKNOWN
        if score_max > CONFIDENCE_CUTOFF:
            class_idx = classes[np.argmax(scores)]
            if class_idx == 1:
                state = TrafficLight.GREEN
            elif class_idx == 2:
                state = TrafficLight.YELLOW
            elif class_idx == 3:
                state = TrafficLight.RED
            else:
                state = TrafficLight.UNKNOWN

        return state
