import cv2
import numpy as np
import pytest
from padel_preprocess import sample_frames_to_video

# Dummy VideoCapture to simulate frames
class DummyCapture:
    def __init__(self, frames, fps=30, width=100, height=100):
        self.frames = frames
        self.index = 0
        self.fps = fps
        self.width = width
        self.height = height

    def isOpened(self):
        return True

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

    def set(self, prop, val):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.index = int(val)

    def read(self):
        if self.index < len(self.frames):
            frame = self.frames[self.index]
            self.index += 1
            return True, frame
        return False, None

    def release(self):
        pass

@pytest.fixture(autouse=True)
def patch_videocapture(monkeypatch):
    # Create 3 dummy black frames
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]
    cap = DummyCapture(frames)
    monkeypatch.setattr(cv2, 'VideoCapture', lambda path: cap)

def test_sample_all_frames(tmp_path):
    input_video = 'dummy.mp4'
    output_video = tmp_path / 'out.mp4'
    # Request more frames than exist → should sample all frames
    sample_frames_to_video(str(input_video), str(output_video), num_frames=5)
    assert output_video.exists()

def test_sample_segment_frames(tmp_path):
    input_video = 'dummy.mp4'
    output_video = tmp_path / 'out_seg.mp4'
    # Sample 2 frames from seconds 0–1
    sample_frames_to_video(str(input_video), str(output_video),
                           num_frames=2, start_sec=0.0, end_sec=1.0)
    assert output_video.exists()
