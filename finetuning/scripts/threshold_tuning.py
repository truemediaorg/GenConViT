import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def calculate_metrics(y_true, y_pred):
    
    cm = confusion_matrix(y_true, y_pred, labels=['REAL', 'FAKE'])  # For a binary classification: [[TN, FP], [FN, TP]]
    
    TN, FP, FN, TP = cm.ravel()

    # Print confusion matrix and other metrics
    print("Confusion Matrix:\n", cm)
    print("True Positives (TP):", TP)
    print("True Negatives (TN):", TN)
    print("False Positives (FP):", FP)
    print("False Negatives (FN):", FN)

    print("%TN:", TN / (TN+FP))
    print("precision:", TP / (TP+FP))


    print(classification_report(y_true, y_pred, target_names=['FAKE', 'REAL']))
    print("Accuracy:", accuracy_score(y_true, y_pred))

def threshold_tuning_metrics(csv_file_path, threshold):
    # Load the data from CSV file
    df = pd.read_csv(csv_file_path)

    # Apply threshold to Confidence scores to make new predictions
    df['New_Prediction'] = df['Confidence'].apply(lambda x: 'FAKE' if x > threshold else 'REAL')

    calculate_metrics(df['Label'], df['New_Prediction'])


# Main function
def main():
    csv_file_path = '/home/ubuntu/lin/global-ml/GenConViT/result/video_prediction_results_2024-04-28_22-53-52.csv'
    
    threshold = 0.33

    threshold_tuning_metrics(csv_file_path, threshold)



if __name__ == "__main__":
    main()
