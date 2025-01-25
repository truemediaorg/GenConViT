import os
import argparse
import pandas as pd
from model.pred_func import *
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from datetime import datetime

def eval(
     vae_name, ed_name, root_dir="sample_prediction_data", num_frames=15, net=None, fp16=False, ignored=True
):
    
    model = load_genconvit(net, fp16, ed=ed_name, vae=vae_name)
    results = pd.DataFrame(columns=['Split', 'Label', 'Filename', 'Prediction', 'Confidence'])

    for split in [ "valid_vid", "test_vid", "train_vid"]: 
        data_dir = os.path.join(root_dir, split)
        print(f"showing results for {split}")
        y_labels = []
        y_preds = []
        for correct_label in ["fake", "real"]:
            label_dir = os.path.join(data_dir, correct_label)
            correct_label = correct_label.upper()
            for filename in tqdm(os.listdir(label_dir), total=len(os.listdir(label_dir))):
                curr_vid = os.path.join(label_dir, filename)
                try:
                    if is_video(curr_vid):
                        y_pred, y_val = predict(
                            curr_vid,
                            model,
                            fp16,
                            num_frames,
                            net,
                        )
                        if ignored and y_val != 0.5:
                            y_preds.append(real_or_fake(y_pred))
                            y_labels.append(correct_label)
                            new_row = pd.DataFrame({
                                'Split': [split],               
                                'Label': [correct_label],
                                'Filename': [filename],
                                'Prediction': [real_or_fake(y_pred)],  
                                'Confidence': [y_val]
                            })
                        elif not ignored:
                            y_preds.append(real_or_fake(y_pred))
                            y_labels.append(correct_label)
                            new_row = pd.DataFrame({
                                'Split': [split],               
                                'Label': [correct_label],
                                'Filename': [filename],
                                'Prediction': [real_or_fake(y_pred)],  
                                'Confidence': [y_val]
                            })


                        results = pd.concat([results, new_row], ignore_index=True)

                    else:
                        print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")
                except Exception as e:
                    print(f"An error occurred in processing video {split}/{correct_label}/{curr_vid}: {str(e)}")
        calculate_metrics(y_true=y_labels, y_pred=y_preds)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results.to_csv(f'result/video_prediction_results_{current_time}.csv', index=False)



def eval_all(
     vae_name, ed_name, root_dir="sample_prediction_data", num_frames=15, net=None, fp16=False
):
    
    model = load_genconvit(net, fp16, ed=ed_name, vae=vae_name)
    results = pd.DataFrame(columns=['Label', 'Filename', 'Prediction', 'Confidence'])

    y_labels = []
    y_preds = []
    for correct_label in ["fake", "real"]:
        label_dir = os.path.join(root_dir, correct_label)
        correct_label = correct_label.upper()
        for filename in tqdm(os.listdir(label_dir), total=len(os.listdir(label_dir))):
            curr_vid = os.path.join(label_dir, filename)
            try:
                if is_video(curr_vid):
                    y_pred, y_val = predict(
                        curr_vid,
                        model,
                        fp16,
                        num_frames,
                        net,
                    )
                    # if y_val != 0.5:
                    y_preds.append(real_or_fake(y_pred))
                    y_labels.append(correct_label)
                    new_row = pd.DataFrame({
                        'Label': [correct_label],
                        'Filename': [filename],
                        'Prediction': [real_or_fake(y_pred)],  
                        'Confidence': [y_val]
                    })

                    results = pd.concat([results, new_row], ignore_index=True)
                    # else: 
                    #     print(f"y_val = 0.5 for processing video {correct_label}/{curr_vid}")
                else:
                    print(f"Invalid video file: {curr_vid}. Please provide a valid video file.")
            except Exception as e:
                print(f"An error occurred in processing video {correct_label}/{curr_vid}: {str(e)}")
    calculate_metrics(y_true=y_labels, y_pred=y_preds)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results.to_csv(f'result/video_prediction_results_{current_time}.csv', index=False)


def predict(
    vid,
    model,
    fp16,
    num_frames,
    net
):
    df = df_face(vid, num_frames, net)  # extract face from the frames
    if fp16:
        df.half()
    y, y_val = (
        pred_vid(df, model)
        if len(df) >= 1
        else (torch.tensor(0).item(), torch.tensor(0.5).item())
    )
    return y, y_val


def gen_parser():
    parser = argparse.ArgumentParser("GenConViT benchmark test")
    parser.add_argument("--f", type=int, help="number of frames to process for prediction")
    parser.add_argument("--d", type=str, help="data path")
    parser.add_argument("--n", type=str, choices=['ed', 'vae'], help="network ed or vae")
    parser.add_argument("--fp16", action='store_true', help="half precision support")
    parser.add_argument("--vae", type=str, help="load pretrained vae model")
    parser.add_argument("--ed", type=str, help="load pretrained ed model")
    parser.add_argument("--eval_all", action='store_true', help="evaluation on all data")
    parser.add_argument("--include_unkown", action='store_true', help="bool flag to test if ignore the case y_val = 0.5")

    args = parser.parse_args()
    is_eval_all = True if args.eval_all else False
    num_frames = args.f if args.f else 15
    data_dir = args.d if args.d else "data"
    net = args.n if args.n else "genconvit"
    fp16 = args.fp16
    vae_name = args.vae if args.vae else "genconvit_vae_inference"
    ed_name = args.ed if args.ed else "genconvit_ed_inference"
    ignored = False if args.include_unkown else True

    return data_dir, num_frames, net, fp16, vae_name, ed_name, is_eval_all, ignored


def calculate_metrics(y_true, y_pred):
    print(classification_report(y_true, y_pred, target_names=['FAKE', 'REAL']))
    print("Accuracy:", accuracy_score(y_true, y_pred))

def main():
    data_dir, num_frames, net, fp16, vae_name, ed_name, is_eval_all, ignored = gen_parser()
    if is_eval_all:
        eval_all(vae_name, ed_name, root_dir=data_dir, num_frames=num_frames, net=net, fp16=fp16, ignored=ignored)
    else: 
        eval(vae_name, ed_name, root_dir=data_dir, num_frames=num_frames, net=net, fp16=fp16, ignored=ignored)


if __name__ == "__main__":
    main()