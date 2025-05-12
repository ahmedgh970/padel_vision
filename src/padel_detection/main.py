import argparse
import json
from padel_detection.detector import Detector
from padel_detection.detection_pipeline import process_input
import os

def cli():
    parser = argparse.ArgumentParser(
        description='Run padel match detection on image or video.'
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Input file path (image or video)')
    parser.add_argument('-o', '--output', default='experiments/outputs/detections.json',
                        help='Output JSON file path')
    parser.add_argument('-t', '--threshold', type=float, default=10.0,
                        help='Proximity threshold in pixels')
    args = parser.parse_args()

    detector = Detector()
    detections = process_input(
        path=args.input,
        detector=detector,
        proximity_thresh=args.threshold
    )
    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(args.output, 'w') as f:
        json.dump(detections, f, indent=2)
    print(f'Detections saved to {args.output}')