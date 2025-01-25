# GenConVit Finetuning

This is the folder for the finetuning for GenConVit Model. 

## To download the video 
1. Set up the aws s3 
2. Make the script executable by running `chmod +x copy_videos.sh` in your terminal in video_downloading folder
3. Run the script by typing `./copy_videos.sh` in your terminal.

## Data Preparation for Video Splitting
The `data_preparation.py` script organizes videos into training, validation, and test datasets based on the information in `eval_ids.txt`. It follows a random split ratio of 60% for training, 20% for validation, and 20% for testing.


## Face extraction
The `face_extraction.py` extracts the cropped face from the splitted videos. It expects a 
folder in such structure 
```
data_folder/
│
├── test_vid/
│   ├── real/
│   │   └── video1.mp4
│   └── fake/
│       └── video2.mp4
│
├── train_vid/
│   ├── real/
│   │   └── video3.mp4
│   └── fake/
│       └── video4.mp4
│
└── valid_vid/
    ├── real/
    │   └── video5.mp4
    └── fake/
        └── video6.mp4
```
To run the script, use the following command with the directory path where the videos are stored:
```
python face_extraction.py --d [path to data_folder]
```

## Benchmarking 
`../benchmark.py` evaluates a pretrained model's performance on video data, generating key metrics like F1-score, recall, and precision.

Example usage:
```
python benchmark.py --f 15 --d /path/to/data --n genconvit --fp16 --vae your_vae_model --ed your_ed_model --eval_all --include_unknown
```

Options:
```
--d: [Required] Data directory path.
--f: Number of frames (default: 15).
--n: Network type ('ed' or 'vae', default: 'genconvit').
--fp16: Enables half precision.
--vae: Pretrained VAE model name in the weight folder.
--ed: Pretrained ED model name in the weight folder
--eval_all: Evaluate all data without splitting.
--include_unknown: Include cases with model confidence of 0.5
```

It output metrics to console and saves prediction results in the result folder for further analysis.




