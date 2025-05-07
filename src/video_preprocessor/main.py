import argparse
from .preprocessor import sample_frames_to_video

def cli():
    parser = argparse.ArgumentParser(
        description='Sample frames from a video segment into a new video.'
    )
    parser.add_argument('-i','--input', required=True, help='Input video path')
    parser.add_argument('-o','--output', required=True, help='Output video path')
    parser.add_argument('-n','--num_frames', type=int, default=10, help='Number of frames to sample')
    parser.add_argument('--start_sec', type=float, default=0.0, help='Segment start time in seconds')
    parser.add_argument('--end_sec', type=float, default=None, help='Segment end time in seconds')
    args = parser.parse_args()

    sample_frames_to_video(
        input_path=args.input,
        output_path=args.output,
        num_frames=args.num_frames,
        start_sec=args.start_sec,
        end_sec=args.end_sec
    )
    print(f'Sampled video saved to {args.output}')