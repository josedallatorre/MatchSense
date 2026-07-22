import argparse

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames


parser = argparse.ArgumentParser(
    description='Download data for action spotting.'
)
parser.add_argument('--dataset_dir', type=str, default="data/soccernet/action-spotting-2023",
                    help="Path for dataset directory")
parser.add_argument('--password_videos', type=str, required=True,
                    help="Password to videos from the NDA")
parser.add_argument('--only_train_valid', action='store_true',
                    help="Download only train and valid splits")
args = parser.parse_args()


if __name__ == "__main__":
    if args.only_train_valid:
        list_splits = ["train", "valid"]
    else:
        list_splits = ["train", "valid", "test"]

    soccernet_downloader = SoccerNetDownloader(LocalDirectory=args.dataset_dir)
    soccernet_downloader.password = args.password_videos
    files_to_download = ["Labels-v2.json", "1_224p.mkv","1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"]
    subset = getListGames(split="train")[:10]
    for i,game in enumerate(subset):
        print(f"Downloading {i}{game}...")
        soccernet_downloader.downloadGame(game=game, files=files_to_download)
    subset = getListGames(split="valid")[:10]
    for i,game in enumerate(subset):
        print(f"Downloading {game}...")
        soccernet_downloader.downloadGame(game=game, files=files_to_download)
    print("✅ Download complete!")
    """
    soccernet_downloader.downloadGames(files=["Labels-v2.json"],
                                       split=list_splits)
    """
    if not args.only_train_valid:
        list_splits.append("challenge")
    """
    soccernet_downloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"],
                                       split=list_splits)
    """
