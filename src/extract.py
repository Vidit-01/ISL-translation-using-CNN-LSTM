import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.interpolate import interp1d
from tqdm import tqdm

TARGET_LEN = 64  # normalized sequence length for all videos
KEYPOINT_DIM = 543

def extract_landmarks(result):
    """
    Extract and concatenate all landmark groups into a flat vector from HolisticLandmarkerResult.
    Result structure:
    - pose_landmarks: list of lists of NormalizedLandmark
    - face_landmarks: list of lists of NormalizedLandmark
    - left_hand_landmarks: list of lists of NormalizedLandmark
    - right_hand_landmarks: list of lists of NormalizedLandmark
    """
    
    # Pose: 33 landmarks * 4 (x, y, z, visibility) = 132
    if result.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] 
                         for lm in result.pose_landmarks[0]]).flatten()
    else:
        pose = np.zeros(132)

    # Left hand: 21 landmarks * 3 (x, y, z) = 63
    if result.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] 
                        for lm in result.left_hand_landmarks[0]]).flatten()
    else:
        lh = np.zeros(63)

    # Right hand: 21 landmarks * 3 = 63
    if result.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] 
                        for lm in result.right_hand_landmarks[0]]).flatten()
    else:
        rh = np.zeros(63)

    # Face: Use first 95 face landmarks (lips, eyebrows, nose) = 285 dimensions
    if result.face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z] 
                          for lm in result.face_landmarks[0][:95]]).flatten()
    else:
        face = np.zeros(285)

    return np.concatenate([pose, lh, rh, face])  # (543,)


def extract_keypoints(video_path, model_path):
    """Extract per-frame keypoints from a video file using HolisticLandmarker Task API."""
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_hand_landmarks_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    
    sequence = []
    
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Timestamp in milliseconds
            timestamp_ms = int((frame_idx / fps) * 1000)
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            sequence.append(extract_landmarks(result))
            frame_idx += 1

    cap.release()
    if len(sequence) == 0:
        raise ValueError(f"No frames extracted from: {video_path}")
    return np.array(sequence, dtype=np.float32)  # (T, 543)


def normalize_sequence(seq, target_len=TARGET_LEN):
    """Temporally resample sequence to fixed length via linear interpolation."""
    T = seq.shape[0]
    if T == target_len:
        return seq
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x_old, seq, axis=0, kind='linear')
    return f(x_new).astype(np.float32)  # (target_len, 543)


def preprocess_dataset(video_dir, save_dir, label_map, model_path, split_file=None):
    """
    Extract keypoints for all videos and save as .npy files.
    """
    os.makedirs(save_dir, exist_ok=True)
    failed = []

    if split_file:
        with open(split_file) as f:
            entries = [line.strip().split() for line in f]
        items = [(os.path.join(video_dir, p), int(lbl)) for p, lbl in entries]
    else:
        items = []
        for label_name in os.listdir(video_dir):
            if label_name not in label_map:
                continue
            label_id = label_map[label_name]
            class_dir = os.path.join(video_dir, label_name)
            for vid in os.listdir(class_dir):
                if vid.endswith(('.mp4', '.avi', '.mov')):
                    items.append((os.path.join(class_dir, vid), label_id))

    for video_path, label_id in tqdm(items, desc="Extracting keypoints"):
        vid_name = os.path.splitext(os.path.basename(video_path))[0]
        save_path = os.path.join(save_dir, f"{vid_name}_{label_id}.npy")

        if os.path.exists(save_path):
            continue  # resume-safe

        try:
            kp = extract_keypoints(video_path, model_path)
            kp = normalize_sequence(kp)
            np.save(save_path, kp)
        except Exception as e:
            print(f"FAILED: {video_path} — {e}")
            failed.append(video_path)

    if failed:
        with open(os.path.join(save_dir, "failed.txt"), "w") as f:
            f.write("\n".join(failed))
    print(f"Done. Failed: {len(failed)}/{len(items)}")


def verify_keypoints(save_dir, expected_shape=(64, 543)):
    """Sanity check all saved .npy files."""
    broken = []
    for fname in os.listdir(save_dir):
        if not fname.endswith('.npy'):
            continue
        arr = np.load(os.path.join(save_dir, fname))
        if arr.shape != expected_shape:
            broken.append((fname, arr.shape))
    if broken:
        print(f"Shape mismatches: {broken}")
    else:
        print(f"All {len(os.listdir(save_dir))} files OK, shape {expected_shape}")
