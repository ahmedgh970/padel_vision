import json
import csv
import os
import tempfile
import hydra
from omegaconf import DictConfig
from padel_detect import Detector, process_input
from padel_preprocess import sample_frames_to_video
from padel_pipe import annotate_and_save_frames

@hydra.main(version_base="1.3", config_name="padel_pipe")
def cli(cfg: DictConfig):
    """
    End-to-end PadelVision pipeline: optional sampling, detection, CSV/JSON export, annotated video
    """
    # Optional sampling
    inpt_path = cfg.input.path
    temp_file = None
    if cfg.preprocess.num_frames > 0:
        temp_file = tempfile.NamedTemporaryFile(suffix=os.path.splitext(inpt_path)[1], delete=False).name
        sample_frames_to_video(
            input_path=inpt_path,
            output_path=temp_file,
            num_frames=cfg.preprocess.num_frames,
            start_sec=cfg.preprocess.start_sec,
            end_sec=cfg.preprocess.end_sec
        )
        inpt_path = temp_file

    # Detection
    detector = Detector(
        pose_model_path=cfg.models.pose,
        ball_model_path=cfg.models.ball,
        confidence_thresh=cfg.models.confidence,
    )
    detections = process_input(
        path=inpt_path,
        detector=detector,
        proximity_thresh=cfg.models.proximity,
    )

    # JSON export
    with open(cfg.output.json, 'w') as f:
        json.dump(detections, f, indent=2)
    print(f"JSON exported to {cfg.output.json}")

    # Annotated and save detected frames
    if cfg.output.annot_dir:
        annotate_and_save_frames(inpt_path, detections, cfg.output.annot_dir)
    print(f"Annotated frames saved to {cfg.output.annot_dir}")

    if temp_file:
        os.remove(temp_file)

if __name__ == '__main__':
    cli()
