# evaluate.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score

def evaluate_model(model, test_loader, device, categories):
    model.eval()
    all_preds = []
    all_true = []

    correct_pred = {classname: 0 for classname in categories}
    total_pred = {classname: 0 for classname in categories}

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size, ncrops, c, h, w = inputs.size()

            outputs = model(inputs.view(-1, c, h, w))  # resize as (batch_size * ncrops, c, h, w)
            outputs = outputs.view(batch_size, ncrops, -1).mean(1)  # Calculate the average of each image crop

            _, predictions = torch.max(outputs, 1)

            all_preds.extend(predictions.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[categories[label]] += 1
                total_pred[categories[label]] += 1

    class_metrics = {}
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        class_metrics[classname] = {'Accuracy': accuracy}
        print(f'Accuracy for {classname:5s} is {accuracy:.6f}')

    f1 = f1_score(all_true, all_preds, average=None, labels=list(range(len(categories))))
    recall = recall_score(all_true, all_preds, average=None, labels=list(range(len(categories))))

    for i, classname in enumerate(categories):
        class_metrics[classname]['F1 Score'] = f1[i]
        class_metrics[classname]['Recall'] = recall[i]
        print(f'F1 Score for {classname:5s} is {f1[i]:.6f}, Recall for {classname:5s} is {recall[i]:.6f}')

    overall_f1 = f1_score(all_true, all_preds, average='weighted')
    overall_recall = recall_score(all_true, all_preds, average='weighted')
    print(f'Overall F1 Score: {overall_f1:.6f}')
    print(f'Overall Recall: {overall_recall:.6f}')

    return class_metrics, overall_f1, overall_recall, all_true, all_preds
