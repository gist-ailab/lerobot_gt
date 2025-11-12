import os
import json
import tqdm
import cv2
import shutil

from utils import *

instructions = 'Pick up the cup and place it on the box.'

if __name__ == '__main__':
    target_folder = '../CupToBox2'
    episodes = os.listdir(target_folder)
    # episode = sorted(episodes)
    num_episodes = len(episodes)

    for episode in episodes:
        episode_path = os.path.join(target_folder, episode)
        color_folder_path = os.path.join(target_folder, episode, 'color_frames')

        json_file = os.path.join(episode_path, 'transforms.json')
        with open(json_file, 'r') as f:
            try:
                json_info = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {json_file}: {e}, deleting folder {episode}...")
                shutil.rmtree(episode_path)
                continue
        print(json_info.keys(), episode)
        # print(json_info['frame_timestamps'])
        # Ready for the array
        synced_frames = []

        img_arrays = []
        state_arrays = []
        action_arrays = []

        # RGB timestamps processing
        color_frames_data = sorted([
                (get_nanoseconds(d['timestamp']), d)
                for d in json_info['frame_timestamps']['color_frames']
            ], key=lambda x: x[0])
        color_timestamps = [ts for ts, _ in color_frames_data]

        # State timestamps processing
        franka_states_data = sorted([
            (get_nanoseconds(d['timestamp']), d)
            for d in json_info['frame_timestamps']['franka_states']
            ], key=lambda x: x[0])
        franka_timestamps = [ts for ts, _ in franka_states_data]
        

        for color_ts, color_data in color_frames_data:                
            # 1. 가장 가까운 franka_state 찾기
            # color_ts를 기준으로 franka_timestamps 리스트에서 가장 가까운 시간의 인덱스를 찾습니다.
            closest_franka_idx = find_closest_timestamp_index(color_ts, franka_timestamps)
            # print(len(franka_timestamps), closest_franka_idx)
            # 찾은 인덱스를 사용해 해당 franka_state 데이터를 가져옵니다.
            synced_franka_state = franka_states_data[closest_franka_idx][1]

            # 3. 동기화된 데이터를 하나의 딕셔너리로 묶어 리스트에 추가합니다.
            synced_frames.append({
                'color_data': color_data,
                'franka_state': synced_franka_state
            })
        previous_gripper_open = 1
        for i in range(5, len(synced_frames)-2):
                # 현재 프레임과 다음 프레임 데이터를 가져옵니다.
                current_sync_frame = synced_frames[i]
                next_sync_frame = synced_frames[i+1]

                # --- 1. 데이터 로딩 ---
                # synced_frames에 있는 파일 경로를 사용하여 이미지 로드
                color_image_name = current_sync_frame['color_data']['filename'].split('/')[-1]
                color_image = cv2.imread(os.path.join(color_folder_path, color_image_name))

                # --- 2. 현재 상태(Current State) 및 다음 상태(Next State) 정의 ---
                cur_joint_state = current_sync_frame['franka_state']['data']['joint_positions'][:-1] # remove last dim (duplicated gripper state)
                cur_gripper_state = cur_joint_state[-1]
                # print(len(cur_joint_state))
                
                cur_ee_transform = get_fk_solution(cur_joint_state[:7]) 
                cur_ee_state = transform_to_pos_quat(cur_ee_transform)

                n_joint_state = next_sync_frame['franka_state']['data']['joint_positions'][:-1]
                n_gripper_state = n_joint_state[-1]
                
                n_ee_transform = get_fk_solution(n_joint_state[:7])
                n_ee_state = transform_to_pos_quat(n_ee_transform)
                
                delta_gripper = n_gripper_state - cur_gripper_state
                delta = False
                if delta_gripper < -0.002:  # 작은 노이즈를 무시하기 위해 임계값 사용
                    # 그리퍼가 닫히는 경우
                    gripper_open = 0
                    delta = True
                elif delta_gripper > 0.01 and not delta: # 작은 노이즈를 무시하기 위해 임계값 사용
                    # 그리퍼가 열리는 경우
                    gripper_open = 1
                    delta = True
                else:
                    # 변화가 거의 없는 경우 이전 상태 유지
                    gripper_open = previous_gripper_open
                    delta = False
                previous_gripper_open = gripper_open

                # --- 3. 액션(Action) 계산 --- 이산적 그리퍼 상태)
                delta_pose = calculate_delta_pose(cur_ee_state, n_ee_state)
                action = np.concatenate([delta_pose, [gripper_open]])
                cur_ee_state = np.append(cur_ee_state, [gripper_open])

                synced_frames[i]['action'] = action
                synced_frames[i]['state'] = cur_ee_state
        # exit()

        with open(os.path.join(episode_path, 'instruction.txt'), 'w') as f:
            f.write(instructions)
        with open(os.path.join(episode_path, 'action.txt'), 'w') as f:
            for i in range(5, len(synced_frames)-2):
                action = synced_frames[i]['action']
                action_str = ' '.join(map(str, action))
                f.write(action_str + '\n')

        with open(os.path.join(episode_path, 'state.txt'), 'w') as f:
            for i in range(5, len(synced_frames)-2):
                state = synced_frames[i]['state']
                state_str = ' '.join(map(str, state))
                f.write(state_str + '\n')

        output_video_path = color_folder_path.replace('color_frames', f'temp_video.mp4')
        images_to_video(
                image_folder=color_folder_path, 
                output_path=output_video_path, 
                fps=15,
            )
