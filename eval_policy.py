import logging
import time
import matplotlib.pyplot as plt
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
import numpy as np
import draccus
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flow.modeling_diffusion import DiffusionPolicy
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    init_logging,
    log_say,
)

"""
```shell
python -m lerobot.replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --dataset.repo_id=aliberts/record-test \
    --dataset.episode=2
```
"""
@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 15

@dataclass
class PolicyConfig:
    policy_path: str | Path
    device: str = "cuda"  # Device to run the policy on, e.g. 'cuda' or 'cpu'.

@dataclass
class ReplayConfig:
    dataset: DatasetReplayConfig
    policy: PolicyConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = False


def calc_mse_for_single_trajectory():
    pass
@draccus.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])
    actions = dataset.hf_dataset.select_columns("action")
    # load policy
    policy = DiffusionPolicy.from_pretrained(cfg.policy.policy_path)
    log_say("Loaded policy")
    log_say("Replaying episode", cfg.play_sounds, blocking=True)

    pred_action_across_time = []
    gt_action_across_time = []
    state_across_time = []

    for idx in tqdm(range(dataset.num_frames)):
        start_episode_t = time.perf_counter()
        batch = dataset[idx]
        # move batch to device
        batch = {k: v.to(cfg.policy.device).unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # pop action from batch
        batch.pop("action", None)
        with torch.inference_mode():
            action = policy.select_action(batch)
            print(action, actions[idx]['action'])

        pred_action_across_time.append(action.squeeze(0).cpu().numpy())
        gt_action_across_time.append(actions[idx]['action'].cpu().numpy())
        state_across_time.append(batch['observation.state'].squeeze().cpu().numpy())

        # dt_s = time.perf_counter() - start_episode_t
        # # busy_wait(1 / dataset.fps - dt_s)
        # busy_wait(max(1.0 / dataset.fps - dt_s, 0.0))
    # convert to numpy
    pred_action_across_time = np.array(pred_action_across_time)
    gt_action_across_time = np.array(gt_action_across_time)
    state_across_time = np.array(state_across_time)
    
    # calculate mse
    mse = np.mean((pred_action_across_time - gt_action_across_time) ** 2)
    logging.info(f"MSE for actions: {mse}")
    logging.info(f"Predicted actions shape: {pred_action_across_time.shape}")
    logging.info(f"Ground truth actions shape: {gt_action_across_time.shape}")
    logging.info(f"State shape: {state_across_time.shape}")

    action_dim = pred_action_across_time.shape[1]
    # plot the actions

    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(10, 4 * action_dim + 2))
    plt.subplots_adjust(top=0.92, left=0.1, right=0.96, hspace=0.4)
    
    for i, ax in enumerate(axes):
        ax.plot(state_across_time[:, i], label="state joints", alpha=0.7)
        ax.plot(gt_action_across_time[:, i], label="gt action", linewidth=2)
        ax.plot(pred_action_across_time[:, i], label="pred action", linewidth=2)

        # put a dot every action step
        for j in range(0, len(gt_action_across_time), policy.config.n_action_steps):
            if j == 0:
                ax.plot(j, gt_action_across_time[j, i], "ro", label="inference point", markersize=2)
            else:
                ax.plot(j, gt_action_across_time[j, i], "ro", markersize=4)
        ax.set_title(f"Action dimension {i + 1}")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3)

    plt.show()

if __name__ == "__main__":
    replay()

