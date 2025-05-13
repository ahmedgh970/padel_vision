import cv2
import numpy as np
import pytest
from padel_detect import process_input

# Dummy VideoCapture to simulate video frames
def make_capture(frames, fps=10, width=5, height=5):
    class DummyCapture:
        def __init__(self):
            self.frames = frames
            self.index = 0
            self.fps = fps
            self.width = width
            self.height = height
        def isOpened(self):
            return True
        def read(self):
            if self.index < len(self.frames):
                frame = self.frames[self.index]
                self.index += 1
                return True, frame
            return False, None
        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return len(self.frames)
            if prop == cv2.CAP_PROP_FPS:
                return self.fps
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return self.width
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return self.height
            return 0
        def release(self):
            pass
        def set(self, prop, val):
            if prop == cv2.CAP_PROP_POS_FRAMES:
                self.index = int(val)
    return DummyCapture()

# Dummy detector with predefined outputs
class DummyDetector:
    def __init__(self, keypoints_list, ball_positions):
        self.kplist = keypoints_list
        self.ball_positions = ball_positions
        self.call = 0
    def detect_keypoints(self, image):
        return self.kplist[self.call]
    def detect_ball(self, image):
        pos = self.ball_positions[self.call]
        self.call += 1
        return pos

@pytest.fixture(autouse=True)
def patch_video(monkeypatch):
    # Create two dummy frames
    frames = [np.zeros((5,5,3), dtype=np.uint8) for _ in range(2)]
    cap = make_capture(frames)
    monkeypatch.setattr(cv2, 'VideoCapture', lambda path: cap)


def test_process_input_video_no_interaction():
    det = DummyDetector([[], []], [None, None])
    res = process_input('video.mp4', det, proximity_thresh=5)
    assert res == {0: {}, 1: {}}


def test_process_input_video_with_interaction():
    # One player with keypoints, ball near left wrist
    kpts = np.zeros((17,2))
    kpts[9] = [2,2]  # left wrist
    kpts[10] = [10,10]  # right wrist placeholder
    det = DummyDetector([[kpts], []], [(3,3), None])
    res = process_input('video.mp4', det, proximity_thresh=2)
    assert 0 in res and res[0]
    assert res[0]['player_id'] == 0
    assert res[0]['ball'] == (3,3)


def test_process_input_image(tmp_path):
    # Create a dummy image file
    img_path = tmp_path / 'image.jpg'
    dummy_img = np.zeros((5,5,3), dtype=np.uint8)
    cv2.imwrite(str(img_path), dummy_img)
    # Detector returns one detection
    kpts = np.zeros((17,2))
    kpts[9] = [1,1]
    kpts[10] = [10,10]
    det = DummyDetector([[kpts]], [(1,1)])
    res = process_input(str(img_path), det, proximity_thresh=1)
    assert res == {0: {'player_id': 0, 'keypoints': kpts.tolist(), 'ball': (1,1)}}


def test_invalid_input(tmp_path):
    det = DummyDetector([], [])
    with pytest.raises(ValueError):
        process_input(str(tmp_path / 'nonexistent.png'), det, proximity_thresh=5)
