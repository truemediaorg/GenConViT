import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from numpy import argmax
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot


df = pd.read_csv('internal_eval.csv')

#see how many unknowns in GT
df['GT'].value_counts()
df['genconvit'].value_counts()

# Drop rows where the GT column is 'Unknown'
df = df[df['GT'] != 'Unknown']
df = df.dropna(subset=['genconvit'])


y_true = df['GT'].values
scores = df['genconvit'].values

# Convert categorical labels to binary format
y_true = (y_true == 'Fake').astype(float)


# calculate pr curve
precision, recall, thresholds = precision_recall_curve(y_true, scores)


plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='Precision-Recall curve')

# Calculate F1 scores
f1_scores = 2 * (precision * recall) / (precision + recall)
# Handle division by zero
f1_scores = np.nan_to_num(f1_scores)
# Find the index of the best threshold
best_threshold_index = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_index]

# Add a larger dot for the best threshold.
plt.scatter(recall[best_threshold_index], precision[best_threshold_index], s=100, c='red', label='Best threshold (F1 Score)')

# Annotate the best threshold point
plt.annotate(f'Threshold: {best_threshold:.2f}\nPrecision: {precision[best_threshold_index]:.2f}',
             (recall[best_threshold_index], precision[best_threshold_index]),
             textcoords="offset points", xytext=(10,-15), ha='center')

# Precision values to be highlighted
target_precisions = [0.75, 0.8]
colors = ['green', 'blue']
for target_precision, color in zip(target_precisions, colors):
    # Find the index closest to the target precision
    closest_precision_index = np.argmin(np.abs(precision - target_precision))
    closest_threshold = thresholds[closest_precision_index]
    
    # Plot the corresponding recall and precision as a large dot
    plt.scatter(recall[closest_precision_index], precision[closest_precision_index], s=100, c=color, label=f'Precision {target_precision}')
    
    # Annotate the threshold and precision point
    plt.annotate(f'Threshold: {closest_threshold:.2f}\nPrecision: {precision[closest_precision_index]:.2f}',
                 (recall[closest_precision_index], precision[closest_precision_index]),
                 textcoords="offset points", xytext=(10,10), ha='center')

# Find the index for the threshold closest to 0.3
specific_threshold = 0.35
closest_index = np.argmin(np.abs(thresholds - specific_threshold))
specific_precision = precision[closest_index]
specific_recall = recall[closest_index]

# Plot the corresponding recall and precision as a large dot
plt.scatter(specific_recall, specific_precision, s=100, c='orange', label=f'Threshold 0.3')

# Annotate the threshold and precision point
plt.annotate(f'Threshold: {thresholds[closest_index]:.2f}\nPrecision: {specific_precision:.2f}',
             (specific_recall, specific_precision),
             textcoords="offset points", xytext=(10,10), ha='center')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('GenConVit PR Curve with Specific Points Highlighted')
plt.grid(True)
plt.legend(loc='lower left')  # Adjust legend location to avoid overlap
plt.show()

print(f'Best Threshold: {best_threshold:.2f}')
print(f'Best F1 Score: {f1_scores[best_threshold_index]:.2f}')