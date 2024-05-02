import cv2
import os
import re

def read_timestamps_from_file(file_path):
    timestamps = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.findall(r"\d+\.\d+", line)
            if match:
                timestamps.append(float(match[0]))
    return timestamps


def extract_frames(video_path, timestamps, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video = cv2.VideoCapture(video_path)
    
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_numbers = [int(fps * ts) for ts in timestamps]
    
    frame_idx = 0
    success, frame = video.read()
    
    while success:
        if frame_idx in frame_numbers:
            timestamp_index = frame_numbers.index(frame_idx)
            cv2.imwrite(os.path.join(output_folder, f'frame_at_{timestamps[timestamp_index]:.2f}s.jpg'), frame)
        frame_idx += 1
        success, frame = video.read()

    video.release()
    print("Done extracting frames.")


timestamp_path = './carla/infra/i_timestamp.txt'
timestamps = read_timestamps_from_file(timestamp_path)

video_path = './carla/data_setup-2024-02-13_18.40.39.mp4'
output_folder = './photos/'

extract_frames(video_path, timestamps, output_folder)
