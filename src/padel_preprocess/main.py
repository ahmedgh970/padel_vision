import hydra
from omegaconf import DictConfig
from padel_preprocess import sample_frames_to_video

@hydra.main(version_base="1.3", config_name="padel_preprocess")
def cli(cfg: DictConfig):
    sample_frames_to_video(
        input_path=cfg.input.path,
        output_path=cfg.output.path,
        num_frames=cfg.preprocess.num_frames,
        start_sec=cfg.preprocess.start_sec,
        end_sec=cfg.preprocess.end_sec,
    )
    print(f"Sampled video saved to {cfg.output.path}")
