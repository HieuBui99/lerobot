import os
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .compute_stats import get_feature_stats
from .lerobot_dataset import LeRobotDataset
from .utils import validate_episode_buffer, validate_frame, write_episode, write_episode_stats, write_info


class VideoEncoder:
    def __init__(
        self,
        fps: int = 30,
        vcodec: str = "libx264",
        pix_fmt: str = "yuv420p",
        g: Optional[int] = 2,
        crf: Optional[int] = 23,
        qp: Optional[int] = None,
        fast_decode: int = 0,
    ):
        self.buffer = []
        self.fps = fps
        self.vcodec = vcodec
        self.pix_fmt = pix_fmt
        self.g = g
        self.crf = crf
        self.qp = qp
        self.fast_decode = fast_decode

    def set_buffer(self, frames: List[np.ndarray]) -> None:
        self.buffer = frames

    def clear_buffer(self) -> None:
        self.buffer = []

    def encode_video(self, video_path: Union[str, Path]) -> None:
        raise NotImplementedError("Must be implemented by subclasses")


class FFmpegEncoder(VideoEncoder):
    def __init__(
        self, *args, chunk_size: int = 100, preset: str = "medium", clear_after_encode: bool = True, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.process = None
        self.chunk_size = chunk_size
        self.preset = preset
        self.clear_after_encode = clear_after_encode

        # Encoding status tracking
        self.is_encoding = False
        self.encoding_completed = False
        self.encoding_started_at = None
        self.encoding_finished_at = None
        self.total_frames_encoded = 0
        self.total_chunks_encoded = 0
        self.current_chunk = 0
        self.output_path = None

    def is_encoding_completed(self) -> bool:
        return self.encoding_completed

    def get_encoding_status(self) -> dict:
        status = {
            "is_encoding": self.is_encoding,
            "encoding_completed": self.encoding_completed,
            "total_frames": len(self.buffer),
            "total_frames_encoded": self.total_frames_encoded,
            "total_chunks": (len(self.buffer) + self.chunk_size - 1) // self.chunk_size if self.buffer else 0,
            "chunks_encoded": self.total_chunks_encoded,
            "current_chunk": self.current_chunk,
            "progress_percentage": (self.total_frames_encoded / len(self.buffer) * 100) if self.buffer else 0,
            "output_path": str(self.output_path) if self.output_path else None,
        }

        if self.encoding_started_at:
            status["started_at"] = self.encoding_started_at
            status["elapsed_time"] = (self.encoding_finished_at or time.time()) - self.encoding_started_at
            if self.encoding_finished_at:
                status["finished_at"] = self.encoding_finished_at
                status["encoding_time"] = self.encoding_finished_at - self.encoding_started_at
                if self.output_path and self.output_path.exists():
                    status["file_size"] = os.path.getsize(self.output_path)
                    status["file_size_kb"] = status["file_size"] / 1024

                if self.total_frames_encoded > 0:
                    status["encoding_fps"] = self.total_frames_encoded / status["encoding_time"]

        return status

    def encode_video(self, video_path: Union[str, Path]) -> None:
        if not self.buffer:
            raise ValueError("No frames in buffer to encode")

        # Reset encoding status
        self.is_encoding = True
        self.encoding_completed = False
        self.encoding_started_at = time.time()
        self.encoding_finished_at = None
        self.total_frames_encoded = 0
        self.total_chunks_encoded = 0
        self.current_chunk = 0

        video_path = Path(video_path)
        self.output_path = video_path
        video_path.parent.mkdir(parents=True, exist_ok=True)

        # Get the size of the first image
        height, width = self.buffer[0].shape[:2]
        total_frames = len(self.buffer)

        # Directly construct FFmpeg command
        cmd = ["ffmpeg"]
        cmd.extend(["-f", "rawvideo"])
        cmd.extend(["-vcodec", "rawvideo"])
        cmd.extend(["-s", f"{width}x{height}"])
        cmd.extend(["-pix_fmt", "rgb24"])
        cmd.extend(["-r", str(self.fps)])
        cmd.extend(["-i", "-"])
        cmd.extend(["-an"])
        cmd.extend(["-vcodec", self.vcodec])
        cmd.extend(["-pix_fmt", self.pix_fmt])
        cmd.extend(["-preset", self.preset])

        if self.g is not None:
            cmd.extend(["-g", str(self.g)])

        if "nvenc" in self.vcodec:
            if self.qp is not None:
                cmd.extend(["-qp", str(self.qp)])
        else:
            if self.crf is not None:
                cmd.extend(["-crf", str(self.crf)])
            if self.fast_decode:
                if self.vcodec == "libsvtav1":
                    cmd.extend(["-svtav1-params", f"fast-decode={self.fast_decode}"])
                else:
                    cmd.extend(["-tune", "fastdecode"])

        cmd.extend(["-loglevel", "warning"])
        cmd.extend(["-y"])
        cmd.append(str(video_path))

        # Start FFmpeg process
        try:
            self.process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**9
            )
            for i in range(0, total_frames, self.chunk_size):
                self.current_chunk = i // self.chunk_size + 1
                chunk = self.buffer[i : i + self.chunk_size]

                for j, frame in enumerate(chunk):
                    try:
                        self.process.stdin.write(frame.tobytes())
                        self.total_frames_encoded += 1
                    except BrokenPipeError as e:
                        stderr_output = self.process.stderr.read().decode()
                        self.is_encoding = False
                        raise RuntimeError(f"Error in FFmpeg stream processing: {stderr_output}") from e

                self.process.stdin.flush()
                self.total_chunks_encoded += 1

            self.process.stdin.close()
            self.process.wait(timeout=600)

            stderr = self.process.stderr.read().decode()
            if self.process.returncode != 0:
                self.is_encoding = False
                raise RuntimeError(f"FFmpeg encoding failed (code: {self.process.returncode}): {stderr}")

            if not video_path.exists():
                self.is_encoding = False
                raise OSError(f"Video encoding did not work. File not found: {video_path}")

            self.is_encoding = False
            self.encoding_completed = True
            self.encoding_finished_at = time.time()

        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            self.is_encoding = False
            self.encoding_completed = False
            if self.process:
                self.process.kill()
                self.process.wait()
                self.process = None
            raise
        finally:
            if self.clear_after_encode:
                self.clear_buffer()
            if self.process:
                self.process = None


class LeRobotDatasetWrapper(LeRobotDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoders = {}
        self.total_frame_buffer = None
        self.episode_ranges = []
        self._append_in_progress = False

    def video_encoding(self) -> None:
        video_paths = {}
        total_buffer_index = self._extract_episode_indices(
            self.total_frame_buffer['episode_index']
        )
        for episode_index, (start, end) in enumerate(self.episode_ranges):
            episode_buffer = self._extract_episode_buffer(start, end, episode_index)
            for key, ep in episode_buffer.items():
                if 'observation.images' in key:
                    video_path = self.root / self.meta.get_video_file_path(
                        total_buffer_index[episode_index], key
                    )
                    video_paths[key] = str(video_path)
                    self._create_video(ep, video_path)

    def _extract_episode_buffer(self, start: int, end: int, episode_index: int) -> dict:
        buffer = {}
        for key, value in self.total_frame_buffer.items():
            if isinstance(value, (list, np.ndarray)):
                buffer[key] = value[start:end + 1]
            else:
                buffer[key] = value

        episode_length = end - start + 1
        buffer['size'] = episode_length
        buffer['index'] = np.arange(start, end + 1)
        buffer['episode_index'] = np.full((episode_length,), episode_index)

        return buffer

    def _extract_episode_indices(self, flat_episode_index_list: list[int]) -> list[int]:
        if not flat_episode_index_list:
            return []

        buffer_idx = []
        for idx in flat_episode_index_list:
            if not buffer_idx or idx != buffer_idx[-1]:
                buffer_idx.append(idx)
        return buffer_idx

    def append_episode_buffer(self, episode_buffer: dict, episode_length) -> None:
        self._append_in_progress = True

        try:
            if not hasattr(self, 'total_frame_buffer') or self.total_frame_buffer is None:
                self.total_frame_buffer = self.create_episode_buffer()
            if not hasattr(self, 'episode_ranges'):
                self.episode_ranges = []

            start_index = self.total_frame_buffer['size']
            num_new_frames = episode_length
            end_index = start_index + num_new_frames - 1

            for key, value in episode_buffer.items():
                if key == 'size':
                    continue
                if key not in self.total_frame_buffer:
                    if isinstance(value, list):
                        self.total_frame_buffer[key] = value.copy()
                    elif hasattr(value, 'tolist'):
                        self.total_frame_buffer[key] = value.tolist()
                    else:
                        self.total_frame_buffer[key] = [value]
                else:
                    if isinstance(self.total_frame_buffer[key], list):
                        if isinstance(value, list):
                            self.total_frame_buffer[key].extend(value)
                        elif hasattr(value, 'tolist'):
                            self.total_frame_buffer[key].extend(value.tolist())
                        else:
                            self.total_frame_buffer[key].append(value)

            if (
                'episode_index' not in self.total_frame_buffer
                or not isinstance(self.total_frame_buffer['episode_index'], list)
            ):
                self.total_frame_buffer['episode_index'] = []

            ep_idx = episode_buffer.get('episode_index')
            if ep_idx is None:
                ep_idx = list(range(episode_length))
            self.total_frame_buffer['episode_index'].extend(ep_idx)

            if 'frame_index' not in episode_buffer:
                self.total_frame_buffer['frame_index'].extend(
                    list(range(start_index, start_index + num_new_frames))
                )

            if 'timestamp' not in episode_buffer:
                self.total_frame_buffer['timestamp'].extend(
                    [(start_index + i) / self.fps for i in range(num_new_frames)]
                )

            self.total_frame_buffer['size'] += num_new_frames

            self.episode_ranges.append((start_index, end_index))
        finally:
            self._append_in_progress = False

    def add_frame_without_write_image(self, frame: dict, task: str) -> None:
        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer['size']
        timestamp = frame.pop('timestamp') if 'timestamp' in frame else frame_index / self.fps
        self.episode_buffer['frame_index'].append(frame_index)
        self.episode_buffer['timestamp'].append(timestamp)

        # Add frame features to episode_buffer
        for key in frame:
            if key not in self.episode_buffer:
                self.episode_buffer[key] = [frame[key]]
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer['task'].append(task)
        self.episode_buffer['size'] += 1

    def save_episode_without_video_encoding(self):
        episode_buffer = self.episode_buffer
        validate_episode_buffer(
            episode_buffer,
            self.meta.total_episodes,
            self.features)

        episode_length = episode_buffer.pop('size')
        tasks = episode_buffer.pop('task')
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer['episode_index']

        episode_buffer['index'] = np.arange(
            self.meta.total_frames,
            self.meta.total_frames + episode_length)
        episode_buffer['episode_index'] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer['task_index'] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            if (key in ['index', 'episode_index', 'task_index'] or
                    ft['dtype'] in ['image', 'video']):
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = self.compute_episode_stats_buffer(episode_buffer, self.features)

        video_paths = {}
        video_count = 0
        for key, ep in self.episode_buffer.items():
            if 'observation.images' in key:
                video_path = self.root / self.meta.get_video_file_path(episode_index, key)
                video_paths[key] = str(video_path)
                video_count += 1
                video_info = {
                    'video.height': self.features[key]['shape'][0],
                    'video.width': self.features[key]['shape'][1],
                    'video.channels': self.features[key]['shape'][2],
                    'video.codec': 'libx264',
                    'video.pix_fmt': 'yuv420p',
                }
                self.meta.info['features'][key]['info'] = video_info

        self.save_meta_info(
            video_count,
            episode_index,
            episode_length,
            episode_tasks,
            ep_stats
        )
        self.append_episode_buffer(episode_buffer, episode_length)

    def save_episode_without_write_image(self):
        episode_buffer = self.episode_buffer
        validate_episode_buffer(
            episode_buffer,
            self.meta.total_episodes,
            self.features)

        episode_length = episode_buffer.pop('size')
        tasks = episode_buffer.pop('task')
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer['episode_index']

        episode_buffer['index'] = np.arange(
            self.meta.total_frames,
            self.meta.total_frames + episode_length)
        episode_buffer['episode_index'] = np.full((episode_length,), episode_index)
        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer['task_index'] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            if (key in ['index', 'episode_index', 'task_index'] or
                    ft['dtype'] in ['image', 'video']):
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = self.compute_episode_stats_buffer(episode_buffer, self.features)

        video_paths = {}
        video_count = 0
        for key, ep in self.episode_buffer.items():
            if 'observation.images' in key:
                video_path = self.root / self.meta.get_video_file_path(episode_index, key)
                video_paths[key] = str(video_path)
                self._create_video(ep, video_path)
                video_count += 1
                video_info = {
                    'video.height': self.features[key]['shape'][0],
                    'video.width': self.features[key]['shape'][1],
                    'video.channels': self.features[key]['shape'][2],
                    'video.codec': 'libx264',
                    'video.pix_fmt': 'yuv420p',
                }
                self.meta.info['features'][key]['info'] = video_info

        self.save_meta_info(
            video_count,
            episode_index,
            episode_length,
            episode_tasks,
            ep_stats
        )

    def save_meta_info(
            self,
            video_count,
            episode_index,
            episode_length,
            episode_tasks,
            episode_stats):
        chunk = self.meta.get_episode_chunk(episode_index)
        if chunk >= self.meta.total_chunks:
            self.meta.info['total_chunks'] += 1
        self.meta.info['total_episodes'] += 1
        self.meta.info['total_frames'] += episode_length
        self.meta.info['total_videos'] += video_count
        self.meta.info['splits'] = {'train': f"0:{self.meta.info['total_episodes']}"}
        self.meta.info['robot_type'] = 'aiworker'

        episode_dict = {
            'episode_index': episode_index,
            'tasks': episode_tasks,
            'length': episode_length,
        }

        write_info(self.meta.info, self.meta.root)
        write_episode(episode_dict, self.meta.root)
        write_episode_stats(episode_index, episode_stats, self.meta.root)

    def _create_video(
            self,
            image_buffer: list[np.ndarray],
            save_path: str):
        if not hasattr(self, 'encoders') or self.encoders is None:
            self.encoders = {}

        self.encoders[save_path] = FFmpegEncoder(
                fps=self.fps,
                chunk_size=50,
                preset='ultrafast',
                crf=28,
                pix_fmt='yuv420p',
                vcodec='libx264'
            )
        self.encoders[save_path].set_buffer(image_buffer)
        self.encoders[save_path].encode_video(save_path)
        # encoding_thread = threading.Thread(
        #     target=self.encoders[save_path].encode_video,
        #     args=(save_path,)
        # )
        # encoding_thread.start()

    def check_video_encoding_completed(self) -> bool:
        if not hasattr(self, 'encoders') or self.encoders is None:
            self.encoders = {}
            return True

        if self.encoders:
            all_completed = True
            completed_encoders = []

            for key, encoder in self.encoders.items():
                if not encoder.encoding_completed:
                    all_completed = False
                else:
                    completed_encoders.append(key)

            for key in completed_encoders:
                encoder = self.encoders[key]
                encoder.clear_buffer()
                del self.encoders[key]
                del encoder

            if all_completed:
                self.encoders = {}
                return True
            else:
                return False

        return True

    def check_append_buffer_completed(self) -> bool:
        return not self._append_in_progress

    def compute_episode_stats_buffer(self, episode_buffer, features):
        ep_stats = {}
        for key, data in episode_buffer.items():
            if features[key]['dtype'] == 'string':
                continue
            elif features[key]['dtype'] in ['image', 'video']:
                ep_ft_array = self._sample_images(data)
                axes_to_reduce = (0, 2, 3)
                keepdims = True
            else:
                ep_ft_array = data
                axes_to_reduce = 0
                keepdims = ep_ft_array.ndim == 1
            ep_stats[key] = get_feature_stats(
                ep_ft_array,
                axis=axes_to_reduce,
                keepdims=keepdims)
            if features[key]['dtype'] in ['image', 'video']:
                ep_stats[key] = {
                    k: v if k == 'count' else np.squeeze(
                        v / 255.0, axis=0) for k, v in ep_stats[key].items()
                }
        return ep_stats

    def _estimate_num_samples(
        self,
        dataset_len: int,
        min_num_samples: int = 100,
        max_num_samples: int = 10_000,
        power: float = 0.75
    ) -> int:
        if dataset_len < min_num_samples:
            min_num_samples = dataset_len
        return max(min_num_samples, min(int(dataset_len**power), max_num_samples))

    def _sample_indices(self, data_len: int) -> list[int]:
        num_samples = self._estimate_num_samples(data_len)
        return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()

    def _auto_downsample_height_width(
            self,
            img: np.ndarray,
            target_size: int = 150,
            max_size_threshold: int = 300):
        _, h, w = img.shape

        if max(w, h) < max_size_threshold:
            return img

        downsample_factor = int(w / target_size) if w > h else int(h / target_size)
        return img[:, ::downsample_factor, ::downsample_factor]

    def _sample_images(self, image_array) -> np.ndarray:
        sampled_indices = self._sample_indices(len(image_array))
        images = None
        for i, idx in enumerate(sampled_indices):
            img = image_array[idx]
            img = np.transpose(img, (2, 0, 1))
            img = self._auto_downsample_height_width(img)
            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)
            images[i] = img

        return images
