import bisect
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import re

def center_crop(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize with aspect ratio preserved, then center-crop to target size."""
    target_w, target_h = size
    h, w = img.shape[:2]
    scale = max(target_w / w, target_h / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    y0 = max(0, (new_h - target_h) // 2)
    x0 = max(0, (new_w - target_w) // 2)
    return resized[y0:y0+target_h, x0:x0+target_w]


def calculate_delta_pose(current_ee_pose, next_ee_pose):
        """
        현재 포즈(current)에서 다음 포즈(next)로 변환하는 델타 포즈를 계산합니다.

        Args:
            current_ee_pose (list or np.ndarray): 기준이 되는 현재 포즈 [x, y, z, qx, qy, qz, qw].
            next_ee_pose (list or np.ndarray): 목표가 되는 다음 포즈 [x, y, z, qx, qy, qz, qw].

        Returns:
            np.ndarray: 현재 포즈에서 다음 포즈로 가는 데 필요한 델타 포즈 [dx, dy, dz, qx, qy, qz, qw].
        """
        # 현재 위치와 다음 위치를 추출합니다.
        current_pos = np.array(current_ee_pose[:3])
        next_pos = np.array(next_ee_pose[:3])

        # 현재 회전과 다음 회전을 추출합니다.
        current_rot = Rotation.from_quat(current_ee_pose[3:])
        next_rot = Rotation.from_quat(next_ee_pose[3:])

        # 1. 델타 회전(delta_rot)을 계산합니다.
        # R_next = R_current * R_delta  --->  R_delta = R_current_inverse * R_next
        # 현재 회전의 역(inverse)을 다음 회전에 곱해줍니다.
        delta_rot = current_rot.inv() * next_rot

        # 2. 델타 위치(delta_pos)를 계산합니다.
        # P_next = P_current + R_current * P_delta  --->  P_delta = R_current_inverse * (P_next - P_current)
        # 두 위치의 차이를 구한 뒤, 이를 현재 회전의 역(inverse)으로 회전시켜
        # 현재 포즈의 로컬 좌표계 기준으로 변환합니다.
        delta_pos = current_rot.inv().apply(next_pos - current_pos)

        # 계산된 델타 위치와 델타 회전을 결합하여 반환합니다.
        delta_pose = np.concatenate((delta_pos, delta_rot.as_quat()))
        
        return delta_pose
        

def get_tf_mat(i, dh):
    a = dh[i][0]
    d = dh[i][1]
    alpha = dh[i][2]
    theta = dh[i][3]
    q = theta

    return np.array([[np.cos(q), -np.sin(q), 0, a],
                     [np.sin(q) * np.cos(alpha), np.cos(q) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
                     [np.sin(q) * np.sin(alpha), np.cos(q) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                     [0, 0, 0, 1]])


def get_fk_solution(joint_angles):
    dh_params = [[0, 0.333, 0, joint_angles[0]],
                 [0, 0, -np.pi/2, joint_angles[1]],
                 [0, 0.316, np.pi/2, joint_angles[2]],
                 [0.0825, 0, np.pi/2, joint_angles[3]],
                 [-0.0825, 0.384, -np.pi/2, joint_angles[4]],
                 [0, 0, np.pi/2, joint_angles[5]],
                 [0.088, 0, np.pi/2, joint_angles[6]],
                 [0, 0.107, 0, 0],
                 [0, 0, 0, -np.pi/4],
                 [0.0, 0.1034, 0, 0]]

    T = np.eye(4)
    for i in range(7 + 3):
        T = T @ get_tf_mat(i, dh_params)
    return T

def get_fk_solution_all(joint_angles):
    dh_params = [[0, 0.333, 0, joint_angles[0]],
                 [0, 0, -np.pi/2, joint_angles[1]],
                 [0, 0.316, np.pi/2, joint_angles[2]],
                 [0.0825, 0, np.pi/2, joint_angles[3]],
                 [-0.0825, 0.384, -np.pi/2, joint_angles[4]],
                 [0, 0, np.pi/2, joint_angles[5]],
                 [0.088, 0, np.pi/2, joint_angles[6]],
                 [0, 0.107, 0, 0],
                 [0, 0, 0, -np.pi/4],
                 [0.0, 0.1034, 0, 0]]

    T = np.eye(4)
    T_list = []
    for i in range(7 + 3):
        T = T @ get_tf_mat(i, dh_params)
        T_list.append(T)
    return T_list

def transform_to_pos_quat(T):
    """
    Converts a 4x4 homogeneous transformation matrix to position and quaternion.
    
    Args:
        T (np.ndarray): The 4x4 transformation matrix.
        
    Returns:
        tuple: A tuple containing:
            - pos (np.ndarray): The (x, y, z) position vector.
            - quat (np.ndarray): The (x, y, z, w) quaternion.
    """
    # 위치(Position)는 변환 행렬의 마지막 열에서 추출합니다.
    pos = T[:3, 3]
    
    # 회전(Rotation)은 변환 행렬의 좌측 상단 3x3 부분 행렬에서 추출합니다.
    R = T[:3, :3]
    
    # 3x3 회전 행렬을 쿼터니언으로 변환합니다.
    # scipy는 [x, y, z, w] 순서로 쿼터니언을 반환합니다.
    quat = Rotation.from_matrix(R).as_quat()

    pose = np.concatenate((pos, quat))

    return pose

def get_nanoseconds(timestamp):
    """ROS 스타일의 타임스탬프를 나노초로 변환합니다."""
    return timestamp['secs'] * 1_000_000_000 + timestamp['nsecs']


def find_closest_timestamp_index(target_ts, sorted_timestamps):
    """
    정렬된 타임스탬프 리스트에서 목표 시간에 가장 가까운 값의 인덱스를 찾습니다.
    효율성을 위해 이진 탐색을 사용합니다.
    """
    # target_ts가 정렬된 리스트에 들어갈 위치를 찾습니다.
    insertion_point = bisect.bisect_left(sorted_timestamps, target_ts)

    # 경계값 처리
    if insertion_point == 0:
        return 0
    if insertion_point == len(sorted_timestamps):
        return len(sorted_timestamps) - 1

    # 삽입점 앞뒤의 값과 비교하여 더 가까운 쪽을 선택합니다.
    before_ts = sorted_timestamps[insertion_point - 1]
    after_ts = sorted_timestamps[insertion_point]

    if after_ts - target_ts < target_ts - before_ts:
        return insertion_point
    else:
        return insertion_point - 1

def images_to_video(
    images: np.ndarray,
    out_path: str,
    fps: int = 30,
    rgb: bool = True,
):
    """
    Save a sequence of images as a video.

    Args:
        images: np.ndarray
            - shape (T, H, W, 3) for color, or (T, H, W) for grayscale
            - dtype uint8, values in [0, 255]
        out_path: str
            Output video file path (e.g., "output.mp4").
        fps: int
            Frames per second.
        rgb: bool
            If True, treats color input as RGB and converts to BGR for OpenCV.
            If False, assumes images already BGR.
    """
    if images.ndim not in (3, 4):
        raise ValueError("images must have shape (T,H,W) or (T,H,W,3)")

    # Handle grayscale
    if images.ndim == 3:
        T, H, W = images.shape
        is_color = False
    else:
        T, H, W, C = images.shape
        if C != 3:
            raise ValueError("Color images must have 3 channels (T,H,W,3)")
        is_color = True

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H), isColor=is_color)

    for i in range(T):
        frame = images[i]

        if is_color:
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            # For grayscale, OpenCV expects 2D array when isColor=False
            frame = frame

        writer.write(frame)

    writer.release()
