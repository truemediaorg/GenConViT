import argparse
import json
from time import perf_counter
from datetime import datetime
import os
import torch
from model.pred_func import *
import typing
import requests
import time  # Import the time module
import hashlib

def generate_hashed_filename(url, original_filename):
    # Generate SHA-256 hash of the URL (32 characters long)
    url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
    
    hashed_filename = f"{url_hash}_{original_filename}"
    
    return hashed_filename

def download_file(input_path):
    """
    Download a file from a given URL and save it locally if input_path is a URL.
    If input_path is a local file path and the file exists, skip the download.

    :param input_path: The URL of the file to download or a local file path.
    :return: The local filepath to the downloaded or existing file.
    """
    # Check if input_path is a URL
    if input_path.startswith(('http://', 'https://')):
        # Extract filename from the URL
        # Splits the URL by '/' and get the last part
        filename = input_path.split('/')[-1]

        # Ensure the filename does not contain query parameters if present in the URL
        # Splits the filename by '?' and get the first part
        filename = filename.split('?')[0]

        # Define the local path where the file will be saved
        local_filepath = os.path.join('.', generate_hashed_filename(input_path, filename))

        # Check if file already exists locally
        if os.path.isfile(local_filepath):
            print(f"The file already exists locally: {local_filepath}")
            return local_filepath

        # Start timing the download
        start_time = time.time()

        # Send a GET request to the URL
        response = requests.get(input_path, stream=True)

        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Open the local file in write-binary mode
        with open(local_filepath, 'wb') as file:
            # Write the content of the response to the local file
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        # End timing the download
        end_time = time.time()

        # Calculate the download duration
        download_duration = end_time - start_time

        print(
            f"Downloaded file saved to {local_filepath} in {download_duration:.2f} seconds.")

    else:
        # Assume input_path is a local file path
        local_filepath = input_path
        # Check if the specified local file exists
        if not os.path.isfile(local_filepath):
            raise FileNotFoundError(f"No such file: '{local_filepath}'")
        print(f"Using existing file: {local_filepath}")

    return local_filepath

class CustomModel:
    """Wrapper for a GenConvit model."""

    def __init__(self, net='genconvit', num_frames=15, fp16=False):
        self.net = net
        self.num_frames = num_frames
        self.fp16 = fp16
        self.model = load_genconvit(net, fp16)
        print("The model is successfully loaded")

    def _predict(self,
                 vid,
                 fp16,
                 result,
                 num_frames,
                 net,
                 klass,
                 count=0,
                 accuracy=-1,
                 correct_label="unknown",
                 compression=None,
                 ):
        count += 1
        print(f"\n\n{str(count)} Loading... {vid}")

        df = df_face(vid, num_frames, net)  # extract face from the frames
        if fp16:
            df.half()
        y, y_val = (
            pred_vid(df, self.model)
            if len(df) >= 1
            else (torch.tensor(0).item(), torch.tensor(0.5).item())
        )

        if accuracy > -1:
            if correct_label == real_or_fake(y):
                accuracy += 1
            print(
                f"\nPrediction: {y_val} {real_or_fake(y)} \t\t {accuracy}/{count} {accuracy/count}"
            )

        return accuracy, count, [y, y_val]

    def predict(self, inputs: typing.Dict[str, str]) -> typing.Dict[str, str]:
        file_path = inputs.get('file_path', None)
        fp16 = inputs.get('fp16', False)
        num_frames = inputs.get('num_frames', 15)
        net = inputs.get('net', "vae")
        video_file = download_file(file_path)
        dataset = None
        result = set_result()
        count = 0

        if os.path.isfile(video_file):
            try:
                if is_video(video_file):
                    accuracy, count, pred = self._predict(
                        video_file,
                        fp16,
                        result,
                        num_frames,
                        net,
                        "uncategorized",
                        count,
                    )
                    print(f"{self.net} is being run.")
                    return {
                        "df_probability": pred[1], "prediction": real_or_fake_thres(pred[1])}

                else:
                    print(
                        f"Invalid video file: {video_file}. Please provide a valid video file.")
            except Exception as e:
                print(f"An error occurred: {str(e)}")
        else:
            print(f"The file {video_file} does not exist.")
        return

    @classmethod
    def fetch(cls) -> None:
        cls()


def main():
    """Entry point for interacting with this model via CLI."""
    start_time = perf_counter()
    curr_time = datetime.now().strftime("%B_%d_%Y_%H_%M_%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("-p", "--file_path",
                        help="The file path for the video file to predict on", required=True, default="https://www.evalai.org/ocasio.mp4")
    parser.add_argument("-f", "--num_frames", type=int, default=15,
                        help="The number of frames to use for prediction")
    parser.add_argument("-n", "--net", type=str, default="vae",
                        help="network ed or vae")
    parser.add_argument("-fp16", action="store_true", default=False,
                        help="Enable FP16 model precision")

    args = parser.parse_args()

    if args.fetch:
        CustomModel.fetch()

    # Create an instance of CustomModel using the arguments
    model = CustomModel(
        net=args.net, num_frames=args.num_frames, fp16=args.fp16)

    # Create inputs dictionary for prediction
    inputs = {
        "file_path": args.file_path,
        "fp16": args.fp16,
        "num_frames": args.num_frames,
        "net": args.net
    }
    # Call predict on the model instance with the specified arguments
    predictions = model.predict(inputs)

    # Optionally, print the predictions if you want to display them
    print(predictions)

    end_time = perf_counter()
    print("\n\n--- %s seconds ---" % (end_time - start_time))


if __name__ == "__main__":
    main()
