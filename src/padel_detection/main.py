import argparse
import json
from padel_detection.detector import Detector
from padel_detection.video_processor import process_video

def cli():
    parser = argparse.ArgumentParser(
        description='Run padel match detection on a video.'
    )
    parser.add_argument(
        '-v', '--video', required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '-o', '--output', default='detections.json',
        help='Path to output JSON file'
    )
    parser.add_argument(
        '-t', '--threshold', type=float, default=10.0,
        help='Pixel threshold for ball-hand proximity'
    )
    args = parser.parse_args()

    detector = Detector()
    detections = process_video(
        path=args.video,
        detector=detector,
        proximity_thresh=args.threshold
    )

    with open(args.output, 'w') as f:
        json.dump(detections, f, indent=2)
    print(f'Detections saved to {args.output}')