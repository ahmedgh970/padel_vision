from ultralytics import YOLO
import numpy as np

class Detector:
    def __init__(self,
                 pose_model_path: str = "yolo11n-pose.pt",
                 ball_model_path: str = "yolo11n.pt"):
        """
        Initialize the Detector with pretrained YOLO pose and ball detection models.
        """
        self.pose_model = YOLO(pose_model_path)
        self.ball_model = YOLO(ball_model_path)
        # COCO class ID for 'sports ball'
        self.ball_class = 32

    def detect_keypoints(self, image: np.ndarray):
        """
        Extract 2D skeleton keypoints for each detected person.
        Returns a list of (num_keypoints, 2) arrays.
        """
        results = self.pose_model(image)
        all_keypoints = []
        # YOLO returns list of results per batch;
        # use first because we are applying the detector on a single image
        r = results[0]
        if hasattr(r, 'keypoints') and r.keypoints is not None:
            kpts_attr = r.keypoints.xy
            # Handle Tensor or NumPy array
            if hasattr(kpts_attr, 'cpu'):
                kpts = kpts_attr.cpu().numpy()
            else:
                kpts = np.array(kpts_attr)
            for person_kpts in kpts:
                all_keypoints.append(person_kpts)
        return all_keypoints

    def detect_ball(self, image: np.ndarray, conf_threshold: float = 0.25):
        """
        Detect the padel ball and return its center (x, y), or None if no valid sports ball detected.
        Selects the detection with the highest confidence above a threshold, and prefers smaller boxes to avoid shadows.
        :param image: BGR image
        :param conf_threshold: Minimum confidence to consider
        """
        results = self.ball_model(image)
        if not results:
            return None
        r = results[0]
        # Extract tensors for boxes, classes, and confidences
        coords_tensor = r.boxes.xyxy
        cls_tensor = r.boxes.cls
        conf_tensor = r.boxes.conf
        # Convert to numpy arrays
        if hasattr(coords_tensor, 'cpu'):
            coords_np = coords_tensor.cpu().numpy()
            cls_np = cls_tensor.cpu().numpy()
            conf_np = conf_tensor.cpu().numpy()
        else:
            coords_np = np.array(coords_tensor)
            cls_np = np.array(cls_tensor)
            conf_np = np.array(conf_tensor)
        # Collect valid detections
        valid = []
        for coords, cls_id, conf in zip(coords_np, cls_np, conf_np):
            if int(cls_id) == int(self.ball_class) and conf >= conf_threshold:
                x1, y1, x2, y2 = coords
                area = (x2 - x1) * (y2 - y1)
                # We use negative area so that smaller boxes rank higher when confidences tie
                valid.append((conf, -area, coords))
        if not valid:
            return None
        # Choose detection with highest confidence, then smallest area
        _, _, best_coords = max(valid, key=lambda x: (x[0], x[1]))
        x1, y1, x2, y2 = best_coords
        cx = float((x1 + x2) / 2)
        cy = float((y1 + y2) / 2)
        return (cx, cy)