import os
import argparse
import json
from time import perf_counter
from datetime import datetime
from model.pred_func import *


def vid(
    file_path, dataset=None, num_frames=15, net=None, fp16=False
):
    result = set_result()
    r = 0
    f = 0
    count = 0
    model = load_genconvit(net, fp16)

    if os.path.isfile(file_path):
        try:
            if is_video(file_path):
                result, accuracy, count, pred = predict(
                    file_path,
                    model,
                    fp16,
                    result,
                    num_frames,
                    net,
                    "uncategorized",
                    count,
                )
                f, r = (
                    f + 1, r) if "FAKE" == real_or_fake(pred[0]) else (f, r + 1)
                print(
                    f"Prediction: {pred[1]} {real_or_fake(pred[0])} \t\tFake: {f} Real: {r}"
                )
            else:
                print(
                    f"Invalid video file: {file_path}. Please provide a valid video file.")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
    else:
        print(f"The file {file_path} does not exist.")

    return result


def gen_parser():
    parser = argparse.ArgumentParser("GenConViT prediction")
    parser.add_argument("--p", type=str, required=False,
                        default=None, help="video or image file path")
    parser.add_argument(
        "--f", type=int, default=15, help="number of frames to process for prediction"
    )
    parser.add_argument(
        "--d", type=str, default="other", help="dataset type, dfdc, faceforensics, timit, celeb"
    )
    parser.add_argument("--n", type=str, default="genconvit",
                        help="network ed or vae")
    parser.add_argument("--fp16", action="store_true",
                        help="half precision support")

    args = parser.parse_args()
    return args.p, args.d, args.f, args.n, args.fp16


def predict(
    vid,
    model,
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
        pred_vid(df, model)
        if len(df) >= 1
        else (torch.tensor(0).item(), torch.tensor(0.5).item())
    )
    result = store_result(
        result, os.path.basename(
            vid), y, y_val, klass, correct_label, compression
    )

    if accuracy > -1:
        if correct_label == real_or_fake(y):
            accuracy += 1
        print(
            f"\nPrediction: {y_val} {real_or_fake(y)} \t\t {accuracy}/{count} {accuracy/count}"
        )

    return result, accuracy, count, [y, y_val]


def main():
    start_time = perf_counter()
    file_path, dataset, num_frames, net, fp16 = gen_parser()
    result = vid(file_path, dataset, num_frames, net, fp16)

    curr_time = datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
    file_name = os.path.basename(file_path).split('.')[0]
    file_path = os.path.join(
        "result", f"prediction_{file_name}_{dataset}_{net}_{curr_time}.json")

    with open(file_path, "w") as f:
        json.dump(result, f)
    end_time = perf_counter()
    print("\n\n--- %s seconds ---" % (end_time - start_time))


if __name__ == "__main__":
    main()
