from PIL import Image
import os
import numpy as np
import shutil
import json
from collections import OrderedDict
import pandas as pd
import cv2

def transform_video_cv2(
    in_path: str,
    out_path: str,
    crop_from_image = (0, 720, 0, 1280),  # (y1, y2, x1, x2) # ori size = y2 = 720, x2 = 1280
    target_size=(256, 256),
    bg_color=(0, 0, 0),
    codec="mp4v"                     # common for .mp4; try "avc1" if needed
):
    def _pad_and_resize(
        frame,
        target_size=(256, 256),
        crop_from_image=(0, 720, 0, 1280),
        bg_color=(0, 0, 0),  # unused now but kept for compatibility
    ):
        """
        Crop the region defined by (top, bottom, left, right) from `frame`,
        then resize it to `target_size` (Wt, Ht) by direct scaling.

        Args:
            frame: np.ndarray, shape (H, W, 3)
            target_size: (Wt, Ht)
            crop_from_image: (top, bottom, left, right) in pixel coords
        """
        H, W = frame.shape[:2]
        top, bottom, left, right = crop_from_image

        # Clamp crop to valid image bounds
        top = max(0, min(H, top))
        bottom = max(0, min(H, bottom))
        left = max(0, min(W, left))
        right = max(0, min(W, right))

        if bottom <= top or right <= left:
            raise ValueError(f"Invalid crop_from_image: {crop_from_image} "
                            f"after clamping to frame size {(H, W)}")

        cropped = frame[top:bottom, left:right]

        Wt, Ht = int(target_size[0]), int(target_size[1])
        resized = cv2.resize(cropped, (Wt, Ht), interpolation=cv2.INTER_AREA)

        return resized
    
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {in_path}")

    # FPS & writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    Wt, Ht = int(target_size[0]), int(target_size[1])
    # print(Wt, Ht)
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (Wt, Ht))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i = 0
    while True:
        ok, frame = cap.read()
        # print(frame.shape)
        if not ok:
            break
        processed = _pad_and_resize(frame, target_size=target_size, crop_from_image=crop_from_image, bg_color=bg_color)
        out.write(processed)
        i += 1

    cap.release()
    out.release()

def stats_from_numpy(array: np.ndarray):
    stats = {
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "min": np.min(array, axis=0).tolist(),
        "q01": np.percentile(array, 1, axis=0).tolist(),
        "q99": np.percentile(array, 99, axis=0).tolist()
    }
    return stats

if __name__ == "__main__":
    data_path = '../../CupToBox_1111/'
    episodes = [x for x in os.listdir(data_path)]
    episodes.sort()


    dataset_save_path = 'CTB_lerobot_dataset'
    default_dataset_path = './gr00t-post-train-dataset-custom/gr1_arms_waist.CanToDrawer'

    all_actions = []
    all_states = []
    # Initialize dataset folder
    if not os.path.exists(f"{dataset_save_path}"):
        os.makedirs(f"{dataset_save_path}")
        os.makedirs(f"{dataset_save_path}/data/chunk-000")
        os.makedirs(f"{dataset_save_path}/meta")
        os.makedirs(f"{dataset_save_path}/videos/chunk-000/observation.images.top_view/")
        os.makedirs(f"{dataset_save_path}/videos/chunk-000/observation.images.side_view/")


    # Copy default dataset meta
    shutil.copy(f"{default_dataset_path}/meta/info.json", f"{dataset_save_path}/meta/info.json")
    shutil.copy(f"{default_dataset_path}/meta/modality.json", f"{dataset_save_path}/meta/modality.json")
    shutil.copy(f"{default_dataset_path}/meta/stats.json", f"{dataset_save_path}/meta/stats.json")

    # Save success video and meta
    last_frame = 0
    for i, episode in enumerate(episodes):
        path = os.path.join(data_path, episode)
        
        video_path1 = os.path.join(path, 'video_top.mp4')
        video_path2 = os.path.join(path, 'video_side.mp4')

        print(video_path1)
        action_path = video_path1.replace('video_top.mp4', 'action.txt')
        state_path = video_path1.replace('video_top.mp4', 'state.txt')
        instruction_path = video_path1.replace('video_top.mp4', 'instruction.txt')

        with open(f"{instruction_path}", 'r') as f:
            instruction = f.read().strip() #.replace("unlocked_waist: ", "")

        with open(f"{action_path}", 'r') as f:
            actions = np.loadtxt(f, dtype=float)
            all_actions.append(actions)

        with open(f"{state_path}", 'r') as f:
            states = np.loadtxt(f, dtype=float)
            all_states.append(states)

        # Copy video
        # shutil.copy(f"{video_path}/{video}", f"{dataset_save_path}/videos/chunk-000/observation.images.ego_view/{episode}.mp4")
        transform_video_cv2(f"{video_path1}", f"{dataset_save_path}/videos/chunk-000/observation.images.top_view/{episode}.mp4", crop_from_image=(0, 720, 280, 1000))
        transform_video_cv2(f"{video_path2}", f"{dataset_save_path}/videos/chunk-000/observation.images.side_view/{episode}.mp4", crop_from_image=(0, 720, 120, 1080))


        # Save meta
        with open(f"{dataset_save_path}/meta/episodes.jsonl", 'w' if i ==0 else 'a') as f:
            episode_info = OrderedDict()
            episode_info["episode_index"] = int(episode[-4:])
            episode_info["tasks"] = instruction
            episode_info["length"] = len(actions)
            episode_info["trajectory_id"] = episode
            episode_info["operator"] = "unknown"
            episode_info["description"] = "GeneratedTrajectory"
            episode_info["remarks"] = instruction
            json.dump(episode_info, f)
            f.write('\n')

        # Save meta
        with open(f"{dataset_save_path}/meta/tasks.jsonl", 'w' if i ==0 else 'a') as f:
            task_info = OrderedDict()
            task_info["task_index"] = i
            task_info["task"] = instruction
            json.dump(task_info, f)
            f.write('\n')

        # Save Parquet
        data = {}
        data['observation.state'] = states.tolist()
        data['action'] = actions.tolist()
        data['timestamp'] = (np.arange(len(actions)) * (1/15)).tolist()
        data['next.reward'] = [0] * len(actions)
        data['next.done'] = [0] * len(actions)
        data['task_index'] = [i+1] * len(actions)
        data['episode_index'] = [i] * len(actions)
        data['index'] = (np.arange(len(actions))+last_frame).tolist()
        data['annotation.human.coarse_action'] = [i] * len(actions)

        df = pd.DataFrame(data)
        df.to_parquet(f"{dataset_save_path}/data/chunk-000/{episode}.parquet", index=False, engine='pyarrow', compression='zstd')

        last_frame += len(actions)

    # Modify the info.json
    with open(f"{dataset_save_path}/meta/info.json", 'r') as f:
        info = json.load(f)
        print(info)
        info['robot_type'] = 'franka_panda'
        info['total_episodes'] = len(episodes)
        info['total_frames'] = last_frame
        info["total_videos"] = len(episodes)
        info["total_chunks"] = 1
        info["total_tasks"] = len(episodes)
        info["fps"] = 15.0
        info["chunk_size"] = len(episodes)

        features = info["features"]
        features['observation.state']= {
                'dtype': "object",
                "shape": [8]
            }
        features['action'] = {
                'dtype': "object",
                "shape": [8]
        }
        observation = features["observation.images.ego_view"]
        video_info = observation["video_info"]
        video_info["video.fps"] = 15.0

        observation["video_info"] = video_info
        features["observation.images.top_view"] = observation
        features["observation.images.side_view"] = observation
        del features["observation.images.ego_view"]
        info["features"] = features

    with open(f"{dataset_save_path}/meta/info.json", 'w') as f:
        json.dump(
            info, f,
            indent=2,            # pretty indent (try 2 or 4)
            ensure_ascii=False,  # keep non-ASCII readable (e.g., Korean)
            sort_keys=False       # optional: deterministic key order
            # separators=(",", ": ")  # default spacing; tweak if needed
        )

    # Modify modality.json
    modality = {
        "state": {
            "joints": {
                "original_key": "observation.state",
                "start": 0,
                "end": 8
            },
        },
        "action": {
            "ee_action": {
                "original_key": "action",
                "start": 0,
                "end": 8
            },
        },
            "video": {
                "top_view": {
                    "original_key": "observation.images.top_view"
                },
                "side_view": {
                    "original_key": "observation.images.side_view"
                }
            },
            "annotation": {
                "human.coarse_action": {
                    "original_key": "annotation.human.coarse_action"
                }
            }
    }
    with open(f"{dataset_save_path}/meta/modality.json", 'w') as f:
        json.dump(
            modality, f,
            indent=2,            # pretty indent (try 2 or 4)
            ensure_ascii=False,  # keep non-ASCII readable (e.g., Korean)
            sort_keys=False       # optional: deterministic key order
            # separators=(",", ": ")  # default spacing; tweak if needed
        )

    actions_concat = np.concatenate(all_actions, axis=0)
    states_concat = np.concatenate(all_states, axis=0)

    actions_stats = stats_from_numpy(actions_concat)
    state_stats = stats_from_numpy(states_concat)

    # Modify stats.json
    with open(f"{dataset_save_path}/meta/stats.json", 'r') as f:
        stats = json.load(f)
        stats['action'] = actions_stats
        stats['observation.state'] = state_stats

    with open(f"{dataset_save_path}/meta/stats.json", 'w') as f:
        json.dump(
            stats, f,
            indent=2,            # pretty indent (try 2 or 4)
            ensure_ascii=False,  # keep non-ASCII readable (e.g., Korean)
            sort_keys=False       # optional: deterministic key order
            # separators=(",", ": ")  # default spacing; tweak if needed
        )