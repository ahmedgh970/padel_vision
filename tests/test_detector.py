import numpy as np
import pytest
from padel_detect import Detector

# Dummy classes to mock YOLO pose and ball models
class DummyPoseModel:
    def __init__(self):
        self.names = {}
    def __call__(self, image):
        class Result:
            def __init__(self):
                # Simulate two people, each with 2 keypoints
                self.keypoints = type('kp', (), {'xy': np.array([[[1,2],[3,4]], [[5,6],[7,8]]])})
        return [Result()]

class DummyBallModel:
    def __init__(self):
        # COCO class ID 32 = 'sports ball'
        self.names = {32: 'sports ball'}
    def __call__(self, image):
        class Result:
            def __init__(self):
                boxes = type('boxes', (), {})()
                # One box at [0,0,2,2]
                boxes.xyxy = np.array([[0, 0, 2, 2]])
                boxes.cls = np.array([32])
                boxes.conf = np.array([0.9])
                self.boxes = boxes
        return [Result()]

@pytest.fixture(autouse=True)
def patch_yolo(monkeypatch):
    # Monkeypatch YOLO to return DummyPoseModel or DummyBallModel
    from padel_detect import detector
    def fake_yolo(path):
        if 'pose' in path:
            return DummyPoseModel()
        return DummyBallModel()
    monkeypatch.setattr(detector, 'YOLO', fake_yolo)


def test_detect_keypoints():
    det = Detector(pose_model_path='pose', ball_model_path='ball')
    kpts_list = det.detect_keypoints(np.zeros((10,10,3), dtype=np.uint8))
    assert isinstance(kpts_list, list)
    assert len(kpts_list) == 2
    assert kpts_list[0].shape == (2,2)


def test_detect_ball_default_threshold():
    det = Detector(pose_model_path='pose', ball_model_path='ball')
    pos = det.detect_ball(np.zeros((10,10,3), dtype=np.uint8))
    # Center of box [0,0,2,2] is (1,1)
    assert pos == (1.0, 1.0)


def test_detect_ball_low_confidence(monkeypatch):
    # Low confidence detection below threshold should return None
    from padel_detect import detector
    class LowConfBallModel(DummyBallModel):
        def __call__(self, image):
            class Result:
                def __init__(self):
                    boxes = type('boxes', (), {})()
                    boxes.xyxy = np.array([[0, 0, 2, 2]])
                    boxes.cls = np.array([32])
                    boxes.conf = np.array([0.1])
                    self.boxes = boxes
            return [Result()]
    def fake_lowconf_yolo(path):
        if 'pose' in path:
            return DummyPoseModel()
        return LowConfBallModel()
    monkeypatch.setattr(detector, 'YOLO', fake_lowconf_yolo)
    det = Detector(pose_model_path='pose', ball_model_path='ball')
    pos = det.detect_ball(np.zeros((10,10,3), dtype=np.uint8))
    assert pos is None