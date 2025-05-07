import numpy as np
import pytest
import cv2
from padel_detection.video_processor import process_video

# Dummy VideoCapture to simulate frames
class DummyCapture:
    def __init__(self, frames):
        self.frames = frames
        self.index = 0
    def read(self):
        if self.index < len(self.frames):
            frame = self.frames[self.index]
            self.index += 1
            return True, frame
        return False, None
    def release(self):
        pass

# Dummy detector with predefined outputs
class DummyDetector:
    def __init__(self, keypoints_list, ball_list):
        self.kplist = keypoints_list
        self.ball_list = ball_list
        self.call = 0
    def detect_keypoints(self, image):
        return self.kplist[self.call]
    def detect_ball(self, image):
        b = self.ball_list[self.call]
        self.call += 1
        return b

@pytest.fixture(autouse=True)
def patch_videocapture(monkeypatch):
    # Create 2 dummy frames
    frames = [np.zeros((5,5,3), dtype=np.uint8) for _ in range(2)]
    cap = DummyCapture(frames)
    monkeypatch.setattr(cv2, 'VideoCapture', lambda path: cap)


def test_process_no_interaction():
    # No interaction: empty keypoints or no ball
    det = DummyDetector([[] , []], [None, None])
    res = process_video('fake.mp4', det, proximity_thresh=10)
    assert res == {0: {}, 1: {}}


def test_process_with_interaction():
    # One frame with one player, ball near wrist
    kpt = np.zeros((11,2))
    kpt[9] = [5,5]  # left wrist
    det = DummyDetector([[kpt], []], [(6,6), None])
    res = process_video('fake.mp4', det, proximity_thresh=2)
    assert 0 in res
    detection = res[0]
    assert detection['player_id'] == 0
    assert detection['ball']['position'] == (6,6)