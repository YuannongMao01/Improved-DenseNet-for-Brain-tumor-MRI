# -*- coding: utf-8 -*-

import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

our_model_name = "se"
confusion_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
models = ['resnet', 'vgg', 'vit', 'densenet', 'efficientnet_v2_m', 'se']
model_official_names = ['ResNet50', "VGG16", "ViT_L_16", "DenseNet121", "Efficient_V2_M", "OurNewModel"]

output_dict = None
with open('final_data_our.json', 'r') as file:
    new_model = json.load(file)

output_dict = None
with open('final_data_baseline.json', 'r') as file:
    output_dict = json.load(file)

output_dict['se'] = new_model['se_improved']

def create_trendline(output_dict, metric_nm="accuracy", dataset="train", model_official_names=None):
    epochs = np.arange(1, 51)
    plt.figure(figsize=(6, 4), dpi=300)  # Suitable size for most papers and high DPI for clarity
    metric = "losses" if metric_nm == "loss" else "acc"
    colors = ['#F28522', '#00CD6C', '#009ADE', '#AF58BA', '#FFC61E', '#FF1F5B']

    if model_official_names is None:
        model_official_names = list(output_dict.keys())

    for i, model in enumerate(output_dict.keys()):
        model_color = colors[i % len(colors)]
        model_metric = output_dict[model][f"{dataset}_{metric}"]
        plt.plot(epochs, model_metric, label=model_official_names[i] if i < 5 else "Our Model", color=model_color,
                linestyle='-', linewidth=1, marker='o', markersize=2)  # Set a uniform line width and marker size

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    plt.legend(fontsize=5)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Epochs', fontsize=8)
    plt.ylabel(metric_nm.capitalize(), fontsize=8)
    plt.title(f"Validation {metric_nm.capitalize()} over Epochs", fontsize=8)
    plt.tight_layout()
    # plt.savefig('/content/4.png')  # Save the figure as a file
    plt.show()

# Train accuracy trendline
create_trendline(output_dict, "accuracy", "train")

# Valid accuracy trendline
create_trendline(output_dict, "accuracy", "valid")

# Train loss trendline
create_trendline(output_dict, "loss", "train")

# Valid loss trendline
create_trendline(output_dict, "loss", "valid")

# Bar plot for accuracy, f1, precision, recall
def category_barplot(category):
    accuracy, f1, precision, recall = [], [], [], []
    for model in models:
        f1_cur = output_dict[model]["by_category_metric"][category]["F1 Score"]
        recall_cur = output_dict[model]["by_category_metric"][category]["Recall"]
        accuracy_cur = output_dict[model]["by_category_metric"][category]["Accuracy"]
        f1_cur = f1_cur if f1_cur <= 1 else f1_cur / 100
        recall_cur = recall_cur if recall_cur <= 1 else recall_cur / 100
        accuracy_cur = accuracy_cur if accuracy_cur <= 1 else accuracy_cur / 100
        precision_cur = f1_cur * recall_cur / (2 * recall_cur - f1_cur)

        accuracy.append(accuracy_cur)
        f1.append(f1_cur)
        precision.append(precision_cur)
        recall.append(recall_cur)

    df = pd.DataFrame({'accuracy': accuracy, 'f1': f1, "precision": precision, "recall": recall}, index = model_official_names)

    ax = df.plot.bar(rot=0, figsize=(20, 3), color = ['#FF1F5B', '#00CD6C', '#009ADE', '#AF58BA'])

    ax.set_ylabel(category.capitalize())
    plt.show()

for category in confusion_labels:
    category_barplot(category)

# Confusion Matrix
conf_matrix = confusion_matrix(output_dict[our_model_name]["true"], output_dict[our_model_name]["preds"])

plt.figure(figsize=(8, 6), dpi=300)
sns.set(font_scale=1)
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
            xticklabels=confusion_labels, yticklabels=confusion_labels,
            cbar_kws={'label': 'Frequency'})

plt.xlabel('Predicted labels', fontsize=14)
plt.ylabel('True labels', fontsize=14)
plt.title(f'Confusion Matrix', fontsize=16)

plt.tight_layout()
plt.show()
# plt.savefig('/content/5.png')
# plt.close()



# chart for related results
df_metrics = pd.DataFrame(data = {
    "accuracy": [accuracy_score(output_dict[model]["true"], output_dict[model]["preds"]) for model in output_dict.keys()],
    "f1_score": [f1_score(output_dict[model]["true"], output_dict[model]["preds"], average = "macro") for model in output_dict.keys()],
    "f1_score (weighted)": [f1_score(output_dict[model]["true"], output_dict[model]["preds"], average = "weighted") for model in output_dict.keys()],
    "precision": [precision_score(output_dict[model]["true"], output_dict[model]["preds"], average = "macro") for model in output_dict.keys()],
    "precision (weighted)": [precision_score(output_dict[model]["true"], output_dict[model]["preds"], average = "weighted") for model in output_dict.keys()],
    "recall": [recall_score(output_dict[model]["true"], output_dict[model]["preds"], average = "macro") for model in output_dict.keys()],
    "recall (weighted)": [recall_score(output_dict[model]["true"], output_dict[model]["preds"], average = "weighted") for model in output_dict.keys()]
}, index = output_dict.keys())

df_metrics

