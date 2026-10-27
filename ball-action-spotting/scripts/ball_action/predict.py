import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np

from src.ball_action.annotations import raw_predictions_to_actions, prepare_game_spotting_results
from src.utils import get_best_model_path, get_video_info
from src.predictors import MultiDimStackerPredictor
from src.frame_fetchers import OpencvFrameFetcher

try:
    from src.frame_fetchers.nvdec import NvDecFrameFetcher
except (ImportError, OSError) as exc:
    print(f"NVDEC unavailable: {exc}. Falling back to OpenCV.")
    NvDecFrameFetcher = None
from src.ball_action import constants


RESOLUTION = "720p"
INDEX_SAVE_ZONE = 1
TTA = True


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument("--folds", default="all", type=str)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--challenge", action="store_true")
    parser.add_argument("--use_saved_predictions", action="store_true")
    parser.add_argument("--model", type=str)
    return parser.parse_args()


def get_raw_predictions(predictor: MultiDimStackerPredictor,
                        video_path: Path,
                        frame_count: int) -> tuple[list[int], np.ndarray]:
    if NvDecFrameFetcher is not None:
        frame_fetcher = NvDecFrameFetcher(
            video_path, gpu_id=predictor.device.index
        )
        print("Using NVIDIA GPU decoder")
    else:
        frame_fetcher = OpencvFrameFetcher(
            video_path, gpu_id=predictor.device.index
        )
        print("Using OpenCV decoder")
    frame_fetcher.num_frames = frame_count

    indexes_generator = predictor.indexes_generator
    min_frame_index = indexes_generator.clip_index(0, frame_count, INDEX_SAVE_ZONE)
    max_frame_index = indexes_generator.clip_index(frame_count, frame_count, INDEX_SAVE_ZONE)
    frame_index2prediction = dict()
    predictor.reset_buffers()
    with tqdm() as t:
        while True:
            frame = frame_fetcher.fetch_frame()
            frame_index = frame_fetcher.current_index
            prediction, predict_index = predictor.predict(frame, frame_index)
            if predict_index < min_frame_index:
                continue
            if prediction is not None:
                frame_index2prediction[predict_index] = prediction.cpu().numpy()
            t.update()
            if predict_index == max_frame_index:
                break
    predictor.reset_buffers()
    frame_indexes = sorted(frame_index2prediction.keys())
    raw_predictions = np.stack([frame_index2prediction[i] for i in frame_indexes], axis=0)
    return frame_indexes, raw_predictions


def predict_video(predictor: MultiDimStackerPredictor,
                  half: int,
                  game_dir: Path,
                  game_prediction_dir: Path,
                  use_saved_predictions: bool) -> dict[str, tuple]:
    video_path = game_dir /  f"{RESOLUTION}.{constants.videos_extension}" #f"{half}_{RESOLUTION}.mkv"
    video_info = get_video_info(video_path)
    print(video_path)
    print("Video info:", video_info)
    assert video_info["fps"] == constants.video_fps

    raw_predictions_path = game_prediction_dir / f"{half}_raw_predictions.npz"
    if use_saved_predictions and raw_predictions_path.exists():
        print(f"Loading saved predictions: {raw_predictions_path}")

        with np.load(str(raw_predictions_path)) as saved:
            frame_indexes = saved["frame_indexes"]
            raw_predictions = saved["raw_predictions"]

    else:
        print("Predict video:", video_path)

        frame_indexes, raw_predictions = get_raw_predictions(
            predictor, video_path, video_info["frame_count"]
        )

        np.savez(
            raw_predictions_path,
            frame_indexes=frame_indexes,
            raw_predictions=raw_predictions,
        )

        print("Raw predictions saved to", raw_predictions_path)

    class2actions = raw_predictions_to_actions(frame_indexes, raw_predictions)
    return class2actions

def predict_game(
    predictor: MultiDimStackerPredictor,
    game: str,
    prediction_dir: Path,
    use_saved_predictions: bool,
):
    game_dir = constants.soccernet_dir / game
    game_prediction_dir = prediction_dir / game
    game_prediction_dir.mkdir(parents=True, exist_ok=True)

    results_path = game_prediction_dir / "results_spotting.json"

    if use_saved_predictions and results_path.exists():
        print(f"Game already completed, skipping: {game}")
        return

    print("Predict game:", game)

    half2class_actions = dict()

    for half in constants.halves:
        class_actions = predict_video(
            predictor,
            half,
            game_dir,
            game_prediction_dir,
            use_saved_predictions,
        )
        half2class_actions[half] = class_actions

    prepare_game_spotting_results(
        half2class_actions,
        game,
        prediction_dir,
    )


def predict_fold(experiment: str, fold, gpu_id: int,
                 challenge: bool, use_saved_predictions: bool, model:str):
    print(f"Predict games: {experiment=}, {fold=}, {gpu_id=} {challenge=}")
    experiment_dir = constants.experiments_dir / experiment / f"fold_{fold}"
    if model is not None:
        model_path = Path(model)
    else:
        model_path = get_best_model_path(experiment_dir)
    print("Model path:", model_path)
    predictor = MultiDimStackerPredictor(model_path, device=f"cuda:{gpu_id}", tta=TTA)
    if challenge:
        data_split = "challenge"
        games = constants.challenge_games
    else:
        if fold == 'train':
            data_split = 'test'
            games = constants.fold2games[5] + constants.fold2games[6]
        else:
            data_split = "cv"
            games = constants.fold2games[fold]
    prediction_dir = constants.predictions_dir / experiment / data_split / f"fold_{fold}"
    if not prediction_dir.exists():
        prediction_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Folder {prediction_dir} already exists.")
    for game in games:
        predict_game(predictor, game, prediction_dir, use_saved_predictions)


if __name__ == "__main__":
    args = parse_arguments()
    if args.folds == "train":
        predict_fold(args.experiment, "train", args.gpu_id,
                     args.challenge, args.use_saved_predictions, args.model)
    else:
        if args.folds == "all":
            folds = constants.folds
        else:
            folds = [int(fold) for fold in args.folds.split(",")]

        for fold in folds:
            predict_fold(args.experiment, fold, args.gpu_id,
                         args.challenge, args.use_saved_predictions, args.model)
