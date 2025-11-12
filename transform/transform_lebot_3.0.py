from PIL import Image
import os
import numpy as np
import shutil
import json
from collections import OrderedDict
import pandas as pd
import cv2
from decord import VideoReader, cpu

def transform_video_array(
    np_input_video: np.ndarray,
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
    
    # print(Wt, Ht)
    out = []
    i = 0
    for frame in np_input_video:
        processed = _pad_and_resize(frame, target_size=target_size, crop_from_image=crop_from_image, bg_color=bg_color)
        out.append(processed)
        i += 1
    out = np.array(out) 
    return out


def save_video_from_numpy(video, path, fps=15):
    """
    video: np.ndarray of shape (T, H, W, 3), RGB uint8
    """
    T, H, W, C = video.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID'
    out = cv2.VideoWriter(path, fourcc, fps, (W, H))

    for i in range(T):
        frame = video[i]
        # convert RGB -> BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

def video_to_numpy(path):
    vr = VideoReader(path, ctx=cpu())
    # all frames: (T, H, W, 3), RGB
    return vr.get_batch(range(len(vr))).asnumpy()

def stats_from_numpy(array: np.ndarray):
    stats = {
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "min": np.min(array, axis=0).tolist(),
        "count": int(array.shape[0]),
        "q01": np.percentile(array, 1, axis=0).tolist(),
        "q99": np.percentile(array, 99, axis=0).tolist()
    }
    return stats

def stats_from_image(array):
    length = array.shape[0]
    array = array.reshape(-1, 3)
    stats = {
        "mean": np.mean(array, axis=0),
        "std": np.std(array, axis=0),
        "max": np.max(array, axis=0),
        "min": np.min(array, axis=0),
        "count": int(length)
    }
    return stats

if __name__ == "__main__":
    data_path = '../../CupToBox_1111/'
    episodes = [x for x in os.listdir(data_path)]
    episodes.sort()

    dataset_save_path = 'CTB_lerobot_dataset_3.0'
    default_dataset_path = './so100'

    all_actions = []
    all_states = []

    # Initialize dataset folder
    if not os.path.exists(f"{dataset_save_path}"):
        os.makedirs(f"{dataset_save_path}")
        os.makedirs(f"{dataset_save_path}/data/chunk-000")
        os.makedirs(f"{dataset_save_path}/meta")
        os.makedirs(f"{dataset_save_path}/meta/episodes/chunk-000")
        os.makedirs(f"{dataset_save_path}/videos/observation.images.top/chunk-000")
        os.makedirs(f"{dataset_save_path}/videos/observation.images.side/chunk-000")


    # Copy default dataset meta
    shutil.copy(f"{default_dataset_path}/meta/info.json", f"{dataset_save_path}/meta/info.json")
    # shutil.copy(f"{default_dataset_path}/meta/modality.json", f"{dataset_save_path}/meta/modality.json")
    shutil.copy(f"{default_dataset_path}/meta/stats.json", f"{dataset_save_path}/meta/stats.json")
    
    # Save success video and meta
    last_frame = 0
    all_data = []
    all_meta = []
    all_task = []
    all_frames1 = []
    all_frames2 = []
    for i, episode in enumerate(episodes):
        path = os.path.join(data_path, episode)
        
        video_path1 = os.path.join(path, 'video_top.mp4')
        video_path2 = os.path.join(path, 'video_side.mp4')

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
        # transform_video_cv2(f"{video_path}", f"{dataset_save_path}/videos/chunk-000/observation.images.ego_view/{episode}.mp4")
    
        video_array1 = video_to_numpy(video_path1)
        video_array1_processed = transform_video_array(video_array1, target_size=(256, 256), crop_from_image=(0, 720, 280, 1000), bg_color=(0,0,0))

        video_array2 = video_to_numpy(video_path2)
        video_array2_processed = transform_video_array(video_array2, target_size=(256, 256), crop_from_image=(0, 720, 120, 1080), bg_color=(0,0,0))

        print(video_array1_processed.shape, video_array2_processed.shape)

        all_frames1.extend(video_array1_processed)
        all_frames2.extend(video_array2_processed)

        # Save meta
        data = {}
        stats = {}
        stats['episode_index'] = i
        stats['data/chunk_index'] = 0
        stats['data/file_index'] = 0
        stats['dataset_from_index'] = last_frame
        stats['dataset_to_index'] = last_frame+len(actions)

        data['episode_index'] = [i] * len(actions)
        data['frame_index'] = np.arange(len(actions)).tolist()
        data['timestamp'] = (np.arange(len(actions)) / 15.).tolist()

        stats['videos/observation.images.top/chunk_index'] = 0
        stats['videos/observation.images.top/file_index'] = 0
        stats['videos/observation.images.top/from_timestamp'] = last_frame/15.0
        stats['videos/observation.images.top/to_timestamp'] = (last_frame+len(actions))/15.0

        stats['videos/observation.images.side/chunk_index'] = 0
        stats['videos/observation.images.side/file_index'] = 0
        stats['videos/observation.images.side/from_timestamp'] = last_frame/15.0
        stats['videos/observation.images.side/to_timestamp'] = (last_frame+len(actions))/15.0

        stats['tasks'] = [instruction]
        stats['length'] = len(actions)

        stat = stats_from_numpy(actions)
        stats['stats/action/min'] = [stat['min']]
        stats['stats/action/max'] = [stat['max']]
        stats['stats/action/mean'] = [stat['mean']]
        stats['stats/action/std'] = [stat['std']]
        stats['stats/action/count'] = [stat['count']]

        stat = stats_from_numpy(states)
        stats['stats/observation.state/min'] = [stat['min']]
        stats['stats/observation.state/max'] = [stat['max']]
        stats['stats/observation.state/mean'] = [stat['mean']]
        stats['stats/observation.state/std'] = [stat['std']]
        stats['stats/observation.state/count'] = [stat['count']]

        stat = stats_from_image(video_array1)
        stats['stats/observation.images.top/min'] = [stat['min']]
        stats['stats/observation.images.top/max'] = [stat['max']]
        stats['stats/observation.images.top/mean'] = [stat['mean']]
        stats['stats/observation.images.top/std'] = [stat['std']]
        stats['stats/observation.images.top/count'] = [stat['count']]

        stat = stats_from_image(video_array2)
        stats['stats/observation.images.side/min'] = [stat['min']]
        stats['stats/observation.images.side/max'] = [stat['max']]
        stats['stats/observation.images.side/mean'] = [stat['mean']]
        stats['stats/observation.images.side/std'] = [stat['std']]
        stats['stats/observation.images.side/count'] = [stat['count']]

        time_stamp = np.array(data['timestamp'])
        stat = stats_from_numpy(time_stamp)
        stats['stats/timestamp/min'] = [stat['min']]
        stats['stats/timestamp/max'] = [stat['max']]
        stats['stats/timestamp/mean'] = [stat['mean']]
        stats['stats/timestamp/std'] = [stat['std']]
        stats['stats/timestamp/count'] = [stat['count']]

        frame_index = np.array(data['frame_index'])
        stat = stats_from_numpy(frame_index)
        stats['stats/frame_index/min'] = [stat['min']]
        stats['stats/frame_index/max'] = [stat['max']]
        stats['stats/frame_index/mean'] = [stat['mean']]
        stats['stats/frame_index/std'] = [stat['std']]
        stats['stats/frame_index/count'] = [stat['count']]


        episode_index = np.array(data['episode_index'])
        stat = stats_from_numpy(episode_index)
        stats['stats/episode_index/min'] = [stat['min']]
        stats['stats/episode_index/max'] = [stat['max']]
        stats['stats/episode_index/mean'] = [stat['mean']]
        stats['stats/episode_index/std'] = [stat['std']]
        stats['stats/episode_index/count'] = [stat['count']]

        index = (np.arange(len(actions))) + last_frame
        stat = stats_from_numpy(index)
        stats['stats/index/min'] = [stat['min']]
        stats['stats/index/max'] = [stat['max']]
        stats['stats/index/mean'] = [stat['mean']]
        stats['stats/index/std'] = [stat['std']]
        stats['stats/index/count'] = [stat['count']]

        task_index = (np.zeros(len(actions)))
        stat = stats_from_numpy(task_index)
        stats['stats/task_index/min'] = [stat['min']]
        stats['stats/task_index/max'] = [stat['max']]
        stats['stats/task_index/mean'] = [stat['mean']]
        stats['stats/task_index/std'] = [stat['std']]
        stats['stats/task_index/count'] = [stat['count']]       
        
        stats['meta/episodes/chunk_index'] = 0
        stats['meta/episodes/file_index'] = 0
        all_meta.append(stats)


        # Save task
        if not instruction in all_task:
            all_task.append(instruction)

        # with open(f"{dataset_save_path}/meta/tasks.jsonl", 'w' if i ==0 else 'a') as f:
        #     task_info = OrderedDict()
        #     task_info["task_index"] = i
        #     task_info["task"] = instruction
        #     json.dump(task_info, f)
        #     f.write('\n')

        # Save Parquet
        data = {}
        data['action'] = actions.tolist()
        data['observation.state'] = states.tolist()
        data['timestamp'] = (np.arange(len(actions)) / 15.).tolist()
        data['frame_index'] = np.arange(len(actions)).tolist()
        data['episode_index'] = [i] * len(actions)
        data['index'] = (np.arange(len(actions))+last_frame).tolist()
        data['task_index'] = [0] * len(actions)

        # data['annotation.human.coarse_action'] = [0] * len(actions)
        last_frame += len(actions)
        all_data.append(data)

    # all_data = all_data
    merged = all_data[0]
    for d in all_data[1:]:
        for k, v in d.items():
            merged[k].extend(v)

    df = pd.DataFrame(merged)
    df.to_parquet(f"{dataset_save_path}/data/chunk-000/file-000.parquet", index=False, engine='pyarrow', compression='zstd')


    task_dict = {}
    for i, instruction in enumerate(all_task):
        task_dict["task_index"] = i
        task_dict["task"] = instruction
    # print(all_task)
    df = pd.DataFrame({"task_index": list(range(len(all_task)))}, index=all_task)
    # df.set_index('task_index', inplace=True)
    df.index = all_task
    df.to_parquet(f"{dataset_save_path}/meta/tasks.parquet", index=True, engine='pyarrow', compression='zstd')



    ## meta file-000.parquet
    df = pd.DataFrame(all_meta)
    df.to_parquet(f"{dataset_save_path}/meta/episodes/chunk-000/file-000.parquet", index=False, engine='pyarrow', compression='zstd')

    # save video
    video_array = np.array(all_frames1)
    save_video_from_numpy(video_array, f"{dataset_save_path}/videos/observation.images.top/chunk-000/file-000.mp4", fps=15)

    video_array = np.array(all_frames2)
    save_video_from_numpy(video_array, f"{dataset_save_path}/videos/observation.images.side/chunk-000/file-000.mp4", fps=15)

    

    # Modify the info.json
    with open(f"{dataset_save_path}/meta/info.json", 'r') as f:
        info = json.load(f)
        # print(info)
        info['robot_type'] = 'franka_panda'
        info['total_episodes'] = len(episodes)
        info['total_frames'] = last_frame
        info["total_videos"] = len(episodes)
        info["total_chunks"] = 1
        info["total_tasks"] = 1
        info["chunk_size"] = 1000
        info["fps"] = 15

        features = info["features"]
        del features['observation.images.wrist']
        features['observation.state']= {
                'dtype': "float32",
                "shape": [8],
                "fps": 15.0,
                "names": [
                    "eex",
                    "eey",
                    "eez",
                    "eer",
                    "eev",
                    "eex",
                    "eeu",
                    "gripper"
                ],
            }
        features['action'] = {
                'dtype': "float32",
                "shape": [8],
                "fps": 15.0,
                "names": [
                    "eex",
                    "eey",
                    "eez",
                    "eer",
                    "eev",
                    "eex",
                    "eeu",
                    "gripper"
                ],
        }
        features['timestamp'] = {
                'dtype': "float32",
                "shape": [1],
                "fps": 15.0,
                "names": "null"
        }
        features['frame_index'] = {
                'dtype': "int64",
                "shape": [1],
                "fps": 15.0,
                "names": "null"
        }
        features['episode_index'] = {
                'dtype': "int64",
                "shape": [1],
                "fps": 15.0,
                "names": "null"
        }
        features['task_index'] = {
                'dtype': "int64",
                "shape": [1],
                "fps": 15.0,
                "names": "null"
        }
        observation = features["observation.images.top"]
        video_info = observation["info"]
        video_info["video.fps"] = 15.0
        video_info["video.height"] = 256
        video_info["video.width"] = 256

        observation["info"] = video_info
        features["observation.images.top"] = observation
        info["features"] = features

    with open(f"{dataset_save_path}/meta/info.json", 'w') as f:
        json.dump(
            info, f,
            indent=4,            # pretty indent (try 2 or 4)
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