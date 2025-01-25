import os
import numpy as np
import cv2
import torch
import dlib
import face_recognition
from decord import VideoReader, cpu
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"


def face_rec(frames):
    cropped_faces = []
    mod = "cnn" if dlib.DLIB_USE_CUDA else "hog"

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        face_locations = face_recognition.face_locations(
            frame, number_of_times_to_upsample=0, model=mod
        )

        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            face_image = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_AREA)
            # face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            cropped_faces.append(face_image)

    return cropped_faces


def extract_frames(video_file, frames_nums=15):
    vr = VideoReader(video_file, ctx=cpu(0))
    step_size = max(1, len(vr) // frames_nums)
    return vr.get_batch(list(range(0, len(vr), step_size))[:frames_nums]).asnumpy()


def df_face(vid, num_frames):
    img = extract_frames(vid, num_frames)
    return face_rec(img)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_faces(faces, directory, base_filename):
    ensure_dir(directory)
    for i, face in enumerate(faces):
        filename = os.path.join(directory, f"{base_filename}_face_{i+1}.png")
        cv2.imwrite(filename, face)
        # print(f"Saved: {filename}")

def process_videos_in_directory(base_dir, num_frames=15):
    for label in ['real','fake']: 
        videos_dir = os.path.join(base_dir + '_vid', label)  # Adjusted to point to *_vid directory for videos
        faces_dir = os.path.join(base_dir, label)  # Faces are saved in a 'faces' subdirectory
        
        # Ensure the faces_dir exists
        ensure_dir(faces_dir)
        
        # Iterate over each video in the directory
        for video_filename in os.listdir(videos_dir):
            print(f"process video {base_dir}/{label}/{video_filename}")
            try:
                if video_filename.endswith(('.mp4', '.avi', '.mov', '.mpeg', '.mpg')):
                    video_path = os.path.join(videos_dir, video_filename)
                    base_filename = os.path.splitext(video_filename)[0]
                
                    
                    # Extract and save faces
                    faces = df_face(video_path, num_frames)
                    save_faces(faces, faces_dir, base_filename)
            except:
                print(f"Error in processing {base_dir}/{label}/{video_filename}")
                

def main(output_base):
    for category in ['train', 'test', 'valid']:
        process_videos_in_directory(os.path.join(output_base, category))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some videos.')
    parser.add_argument('--d', type=str, help='Base directory for output')
    
    args = parser.parse_args()
    main(args.d)