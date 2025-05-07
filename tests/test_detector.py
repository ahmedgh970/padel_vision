import numpy as np
import pytest
from padel_detection.detector import Detector

# Dummy classes to mock YOLO pose and ball models
class DummyPoseModel:
    def __call__(self, image):
        class Result:
            def __init__(self):
                # Simulate two people, each with 2 keypoints
                self.keypoints = type('kp', (), {'xy': np.array([[[1,2],[3,4]], [[5,6],[7,8]]])})
        return [Result()]

class DummyBallModel:
    def __call__(self, image):
        class Result:
            def __init__(self):
                # Simulate one detection box
                self.boxes = type('boxes', (), {'xyxy': np.array([[0,0,2,2]])})
        return [Result()]

@pytest.fixture(autouse=True)
def patch_yolo(monkeypatch):
    # Monkeypatch YOLO constructor to return dummy models based on path
    from padel_detection import detector
    monkeypatch.setattr(detector, 'YOLO', lambda path: DummyPoseModel() if 'pose' in path else DummyBallModel())


def test_detect_keypoints():
    det = Detector(pose_model_path='pose', ball_model_path='ball')
    kpts_list = det.detect_keypoints(np.zeros((10,10,3), dtype=np.uint8))
    assert isinstance(kpts_list, list)
    # Expect two people
    assert len(kpts_list) == 2
    assert kpts_list[0].shape == (2,2)


def test_detect_ball():
    det = Detector(pose_model_path='pose', ball_model_path='ball')
    pos = det.detect_ball(np.zeros((10,10,3), dtype=np.uint8))
    # Center of box [0,0,2,2] -> (1,1)
    assert pos == (1.0, 1.0)