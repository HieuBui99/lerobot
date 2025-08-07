from pathlib import Path
from typing import Callable

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

class EpisodeReader:
    """
    A simple class to read and load all data for a specific episode from a LeRobotDataset.

    Usage:
        episode_reader = EpisodeReader("dataset_repo_id", episode_index=0)
        episode_reader = EpisodeReader(dataset_instance, episode_index=0)
    """

    def __init__(self, dataset: LeRobotDataset | str, episode_index: int):
        """
        Initialize the EpisodeReader for a specific episode.

        Args:
            dataset: Either a LeRobotDataset instance or a repo_id string
            episode_index: The index of the episode to load (0-based)
        """
        # Load the dataset if needed
        if isinstance(dataset, str):
            self.dataset = LeRobotDataset(
                repo_id=dataset,
                episodes=[episode_index],  # Only load the specific episode
            )
        else:
            self.dataset = dataset

        self.episode_index = episode_index
        self._validate_episode_index()

        # Load episode-specific data
        self._load_episode_data()

    def _validate_episode_index(self):
        """Validate that the episode index exists in the dataset."""
        if self.episode_index not in self.dataset.meta.episodes:
            available_episodes = list(self.dataset.meta.episodes.keys())
            raise ValueError(
                f"Episode {self.episode_index} not found in dataset. Available episodes: {available_episodes}"
            )

    def _load_episode_data(self):
        """Load all data for the specific episode."""
        # Get episode metadata
        self.episode_metadata = self.dataset.meta.episodes[self.episode_index]

        # Get episode statistics if available
        self.episode_stats = self.dataset.meta.episodes_stats.get(self.episode_index, {})

        # Find the frame indices that belong to this episode
        if hasattr(self.dataset, "episode_data_index") and self.dataset.episode_data_index is not None:
            episode_start = self.dataset.episode_data_index["from"][self.episode_index].item()
            episode_end = self.dataset.episode_data_index["to"][self.episode_index].item()
            self.frame_indices = list(range(episode_start, episode_end))
        else:
            # Fallback: scan through the dataset to find frames for this episode
            self.frame_indices = []
            for i in range(len(self.dataset)):
                item = self.dataset.hf_dataset[i]
                if item["episode_index"].item() == self.episode_index:
                    self.frame_indices.append(i)

        # Load all frames for this episode
        self.frames = []
        for frame_idx in tqdm(self.frame_indices):
            frame_data = self.dataset[frame_idx]
            self.frames.append(frame_data)

    @property
    def length(self) -> int:
        """Get the number of frames in this episode."""
        return len(self.frames)

    @property
    def tasks(self) -> list[str]:
        """Get the tasks performed in this episode."""
        return self.episode_metadata.get("tasks", [])

    @property
    def duration_s(self) -> float:
        """Get the duration of the episode in seconds."""
        if len(self.frames) == 0:
            return 0.0
        start_time = self.frames[0]["timestamp"].item()
        end_time = self.frames[-1]["timestamp"].item()
        return end_time - start_time

    @property
    def fps(self) -> int:
        """Get the frames per second of the dataset."""
        return self.dataset.fps

    @property
    def features(self) -> dict:
        """Get the features/modalities available in this episode."""
        return self.dataset.features

    @property
    def camera_keys(self) -> list[str]:
        """Get the camera keys available in this episode."""
        return self.dataset.meta.camera_keys

    @property
    def video_keys(self) -> list[str]:
        """Get the video keys available in this episode."""
        return self.dataset.meta.video_keys

    @property
    def image_keys(self) -> list[str]:
        """Get the image keys available in this episode."""
        return self.dataset.meta.image_keys

    def get_frame(self, frame_index: int) -> dict:
        """
        Get a specific frame from the episode.

        Args:
            frame_index: The frame index within the episode (0-based)

        Returns:
            Dictionary containing all data for the specified frame
        """
        if frame_index < 0 or frame_index >= len(self.frames):
            raise IndexError(
                f"Frame index {frame_index} out of range for episode with {len(self.frames)} frames"
            )
        return self.frames[frame_index]

    def get_frames(self, start: int = 0, end: int | None = None) -> list[dict]:
        """
        Get a range of frames from the episode.

        Args:
            start: Starting frame index (inclusive)
            end: Ending frame index (exclusive). If None, goes to the end of the episode

        Returns:
            List of frame dictionaries
        """
        if end is None:
            end = len(self.frames)
        return self.frames[start:end]

    def get_timestamps(self) -> np.ndarray:
        """Get all timestamps for frames in this episode."""
        return np.array([frame["timestamp"].item() for frame in self.frames])

    def get_actions(self) -> np.ndarray:
        """Get all actions for frames in this episode."""
        if "action" not in self.features:
            raise ValueError("No action data available in this dataset")

        actions = []
        for frame in self.frames:
            actions.append(frame["action"].numpy())
        return np.array(actions)

    def get_observations(self, obs_key: str) -> np.ndarray:
        """
        Get observations for a specific observation key.

        Args:
            obs_key: The observation key (e.g., 'observation.state', 'observation.images.cam1')

        Returns:
            Array of observations
        """
        if obs_key not in self.features:
            raise ValueError(f"Observation key '{obs_key}' not available in this dataset")

        observations = []
        for frame in self.frames:
            if obs_key in frame:
                obs_data = frame[obs_key]
                if isinstance(obs_data, torch.Tensor):
                    observations.append(obs_data.numpy())
                else:
                    observations.append(obs_data)

        return np.array(observations)

    def get_camera_images(self, camera_key: str) -> list[np.ndarray]:
        """
        Get all images from a specific camera.

        Args:
            camera_key: The camera key (e.g., 'observation.images.cam1')

        Returns:
            List of image arrays
        """
        if camera_key not in self.camera_keys:
            raise ValueError(
                f"Camera key '{camera_key}' not available. Available cameras: {self.camera_keys}"
            )

        images = []
        for frame in self.frames:
            if camera_key in frame:
                img = frame[camera_key]
                if isinstance(img, torch.Tensor):
                    # Convert from CHW to HWC if needed
                    if len(img.shape) == 3 and img.shape[0] in [1, 3, 4]:  # Likely CHW format
                        img = img.permute(1, 2, 0)
                    images.append(img.numpy())
                else:
                    images.append(img)

        return images

    def get_object_poses(self) -> list[np.ndarray]:
        poses = []
        for frame in self.frames:
            poses.append(frame["object_poses"].numpy())
        return poses

    def iter_frames(self):
        """Iterate over all frames in the episode."""
        for frame in self.frames:
            yield frame

    def save_episode_summary(self, output_path: str | Path) -> None:
        """
        Save a summary of the episode to a file.

        Args:
            output_path: Path where to save the summary
        """
        output_path = Path(output_path)

        summary = {
            "episode_index": self.episode_index,
            "length": self.length,
            "duration_s": self.duration_s,
            "fps": self.fps,
            "tasks": self.tasks,
            "features": list(self.features.keys()),
            "camera_keys": self.camera_keys,
            "video_keys": self.video_keys,
            "image_keys": self.image_keys,
            "episode_metadata": self.episode_metadata,
            "episode_stats": self.episode_stats,
            "timestamps": self.get_timestamps().tolist(),
        }

        import json

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

    def __len__(self) -> int:
        """Get the number of frames in this episode."""
        return self.length

    def __getitem__(self, idx: int) -> dict:
        """Get a frame by index."""
        return self.get_frame(idx)

    def __iter__(self):
        """Iterate over frames in the episode."""
        return iter(self.frames)

    def __repr__(self) -> str:
        return (
            f"EpisodeReader(\n"
            f"  Episode Index: {self.episode_index},\n"
            f"  Length: {self.length} frames,\n"
            f"  Duration: {self.duration_s:.2f}s,\n"
            f"  FPS: {self.fps},\n"
            f"  Tasks: {self.tasks},\n"
            f"  Features: {list(self.features.keys())},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f")"
        )


if __name__ == "__main__":
    # Example usage EpisodeReader(dataset_name, episode_index)
    episode_reader = EpisodeReader("hieu1344/omy_baseline", 0)
    print(f"Episode length: {len(episode_reader)} frames")

    # # Get all observations (follower joint angles)
    obs = episode_reader.get_observations("observation.state")
    print(f"Observation shape: {obs.shape}")

    # # Get all camera images by specifying a camera key
    cam_images = episode_reader.get_camera_images("observation.images.cam_wrist")
    print(f"Number of images from cam wrist: {len(cam_images)}")

    # # Get actions (leader joint angle)
    actions = episode_reader.get_actions()
    print(f"Actions shape: {actions.shape}")

    # get object poses
    object_poses = episode_reader.get_object_poses()
    print(object_poses[0].shape)