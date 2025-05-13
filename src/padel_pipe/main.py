import json
import csv
import os
import tempfile
import hydra
from omegaconf import DictConfig
from padel_detect import Detector, process_input
from padel_preprocess import sample_frames_to_video
from padel_pipe import annotate_video

@hydra.main(version_base="1.3", config_name="padel_pipe")
def cli(cfg: DictConfig):
    """
    End-to-end PadelVision pipeline: optional sampling, detection, CSV/JSON export, annotated video
    """
    # Step 1: Optional sampling
    work_input = cfg.input.path
    temp_file = None
    if cfg.preprocess.num_frames > 0:
        temp_file = tempfile.NamedTemporaryFile(suffix=os.path.splitext(work_input)[1], delete=False).name
        sample_frames_to_video(
            input_path=work_input,
            output_path=temp_file,
            num_frames=cfg.preprocess.num_frames,
            start_sec=cfg.preprocess.start_sec,
            end_sec=cfg.preprocess.end_sec
        )
        work_input = temp_file

    # Step 2: Detection
    detector = Detector(
        pose_model_path=cfg.models.pose,
        ball_model_path=cfg.models.ball,
        conf_threshold=cfg.models.confidence,
    )
    detections = process_input(
        path=work_input,
        detector=detector,
        proximity_thresh=cfg.detect.threshold
    )

    # Step 3: JSON export
    with open(cfg.output.json, 'w') as f:
        json.dump(detections, f, indent=2)
    print(f"JSON exported to {cfg.output.json}")

    # Step 4: CSV export
    with open(cfg.output.csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_id', 'player_id', 'ball_x', 'ball_y'])
        for fid, det in detections.items():
            if det:
                bx, by = det['ball']
                writer.writerow([fid, det['player_id'], bx, by])
    print(f"CSV exported to {cfg.output.csv}")

    # Step 5: Annotated video
    if cfg.output.annotated:
        annotate_video(work_input, detections, cfg.output.annotated)
        print(f"Annotated video saved to {cfg.output.annotated}")

    # Cleanup temp file
    if temp_file:
        os.remove(temp_file)


if __name__ == '__main__':
    cli()
