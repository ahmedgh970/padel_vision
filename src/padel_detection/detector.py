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

    def detect_ball(self, image: np.ndarray):
        """
        Detect the padel ball and return its center (x, y), or None if no sports ball detected.
        """
        results = self.ball_model(image)
        if not results:
            return None
        # YOLO returns list of results per batch; use first because we are using a single image 
        r = results[0]
        # Extract boxes and classes
        coords_tensor = r.boxes.xyxy
        cls_tensor = r.boxes.cls
        # Convert to numpy arrays
        if hasattr(coords_tensor, 'cpu'):
            coords_np = coords_tensor.cpu().numpy()
            cls_np = cls_tensor.cpu().numpy()
        else:
            coords_np = np.array(coords_tensor)
            cls_np = np.array(cls_tensor)
        # Filter for sports ball class
        for coords, cls_id in zip(coords_np, cls_np):
            # TODO: select the ball from the detected onces according to ? 
            if int(cls_id) == int(self.ball_class):
                x1, y1, x2, y2 = coords
                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)
                return (cx, cy)
        return None