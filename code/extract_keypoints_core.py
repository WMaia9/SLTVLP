import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic

# We will use all pose + both hands:
NUM_POSE = 33
NUM_HAND = 21
NUM_JOINTS = NUM_POSE + NUM_HAND * 2  # 33 + 21 + 21 = 75


def _run_mediapipe(video_path: str):
    """Run MediaPipe Holistic on a video and return raw pose/hand tracks."""
    cap = cv2.VideoCapture(video_path)
    pose_list = []
    lh_list = []
    rh_list = []

    with mp_holistic.Holistic(
        model_complexity=2,             # higher quality, a bit slower
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            # BGR -> RGB + mirror, like your old code
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            result = holistic.process(frame)

            # Pose (33 joints)
            if result.pose_landmarks:
                pose = [(lm.x, lm.y, lm.z) for lm in result.pose_landmarks.landmark]
            else:
                pose = [(0.0, 0.0, 0.0)] * NUM_POSE

            # Left hand (21 joints)
            if result.left_hand_landmarks:
                lh = [(lm.x, lm.y, lm.z) for lm in result.left_hand_landmarks.landmark]
            else:
                lh = [(0.0, 0.0, 0.0)] * NUM_HAND

            # Right hand (21 joints)
            if result.right_hand_landmarks:
                rh = [(lm.x, lm.y, lm.z) for lm in result.right_hand_landmarks.landmark]
            else:
                rh = [(0.0, 0.0, 0.0)] * NUM_HAND

            pose_list.append(pose)
            lh_list.append(lh)
            rh_list.append(rh)

    cap.release()
    return pose_list, lh_list, rh_list


def _interpolate_track(track: np.ndarray) -> np.ndarray:
    """
    Interpolate missing joints over time.
    track: (T, J, 3), zeros mean missing.
    Returns (T, J, 3) with gaps filled by linear interpolation or nearest neighbor.
    """
    T, J, C = track.shape
    out = track.copy()

    for j in range(J):
        # mask of valid frames for this joint
        valid = np.any(track[:, j, :] != 0.0, axis=-1)
        if not np.any(valid):
            # joint is missing for all frames, leave as zeros
            continue

        idx = np.arange(T)
        valid_idx = idx[valid]
        for c in range(C):
            vals = track[valid, j, c]
            # interpolate over time
            out[:, j, c] = np.interp(idx, valid_idx, vals)

    return out


def _normalize_skeleton(seq: np.ndarray) -> np.ndarray:
    """
    Normalize skeleton for better model performance:
      - center at mid-hip (or shoulders fallback)
      - scale by shoulder distance
    seq: (T, 75, 3)
    """
    T, J, C = seq.shape
    out = seq.copy()

    # MediaPipe indices:
    # 11: left shoulder, 12: right shoulder, 23: left hip, 24: right hip
    LS, RS, LH, RH = 11, 12, 23, 24

    for t in range(T):
        joints = out[t]

        # choose center: mid-hip if valid, else mid-shoulder, else origin
        center = np.zeros(3, dtype=np.float32)
        if np.any(joints[LH] != 0.0) and np.any(joints[RH] != 0.0):
            center = 0.5 * (joints[LH] + joints[RH])
        elif np.any(joints[LS] != 0.0) and np.any(joints[RS] != 0.0):
            center = 0.5 * (joints[LS] + joints[RS])

        joints = joints - center  # translate

        # compute scale as shoulder distance
        scale = np.linalg.norm(joints[LS] - joints[RS]) if np.any(joints[LS] != 0.0) and np.any(joints[RS] != 0.0) else 0.0
        if scale < 1e-6:
            scale = 1.0

        joints = joints / scale
        out[t] = joints

    return out.astype(np.float32)


def extract_keypoints_best(video_path: str) -> np.ndarray:
    """
    High-quality keypoint extraction for SLT:
      - pose + both hands (75 joints)
      - missing joints interpolated over time
      - skeleton centered & scaled for invariance
    Returns array of shape (T, 75, 3), float32.
    """
    pose_list, lh_list, rh_list = _run_mediapipe(video_path)

    if len(pose_list) == 0:
        return np.zeros((0, NUM_JOINTS, 3), dtype=np.float32)

    T = len(pose_list)
    # Stack into (T, J, 3)
    pose = np.array(pose_list, dtype=np.float32)           # (T, 33, 3)
    lh   = np.array(lh_list,   dtype=np.float32)           # (T, 21, 3)
    rh   = np.array(rh_list,   dtype=np.float32)           # (T, 21, 3)
    all_joints = np.concatenate([pose, lh, rh], axis=1)    # (T, 75, 3)

    # Fill holes
    all_joints = _interpolate_track(all_joints)

    # Normalize
    all_joints = _normalize_skeleton(all_joints)

    return all_joints