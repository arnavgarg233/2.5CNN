import pandas as pd

predictions = pd.read_csv('outputs/main_model/eval/all_predictions.csv', header=None)
labels = pd.read_csv('outputs/main_model/eval/all_labels.csv', header=None)

predictions = predictions.to_numpy()
labels = labels.to_numpy()

for threshold in range(64):
    total_ones = predictions.sum(axis=1, keepdims=True)
    pos_class = total_ones > threshold 
    pos_class = pos_class.astype(int)
    total_correct = (pos_class == labels).sum()
    total_samples = len(predictions)
    accuracy = total_correct / total_samples
    print(f'Threshold: {threshold}, Accuracy: {accuracy:.4f}')




