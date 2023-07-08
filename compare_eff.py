import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from efficientnet_pytorch import EfficientNet

warnings.filterwarnings('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

root_dir = '/home/oyku/texas/val.X/'
categories = os.listdir(root_dir)
mapping_dict = {
    "n01440764": "tench, Tinca tinca",
    "n01443537": "goldfish, Carassius auratus",
    "n01484850": "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
    "n01491361": "tiger shark, Galeocerdo cuvieri",
    "n01494475": "hammerhead, hammerhead shark",
    "n01496331": "electric ray, crampfish, numbfish, torpedo",
    "n01498041": "stingray",
    "n01514668": "cock",
    "n01514859": "hen",
    "n01531178": "goldfinch, Carduelis carduelis"
}

ground_truth_labels = [mapping_dict[category] for category in categories]
data = {}
average_accuracies = []

for model_version in range(8):
    model_name = f'efficientnet-b{model_version}'
    model = EfficientNet.from_pretrained(model_name)
    efficientnet = model.eval().to(device)

    accuracies = []

    for category in categories:
        category_dir = os.path.join(root_dir, category)
        file_paths = os.listdir(category_dir)
        data[category] = file_paths

        ground_truth = mapping_dict[category]
        uris = []
        results = []
        batch_size = 16

        for file_path in file_paths:
            uri = os.path.join(category_dir, file_path)
            uris.append(uri)

        with torch.no_grad():
            for i in range(0, len(uris), batch_size):
                batch_uris = uris[i:i+batch_size]
                batch = torch.cat([utils.prepare_input_from_uri(uri) for uri in batch_uris]).to(device)
                output = torch.nn.functional.softmax(efficientnet(batch), dim=1)
                batch_results = utils.pick_n_best(predictions=output, n=1)
                results.extend(batch_results)

        count = 0
        for result in results:
            if result[0][0] == ground_truth:
                count += 1
            else:
                print("not equal")

        accuracy = count / len(uris) * 100
        accuracies.append(accuracy)

    average_accuracy = sum(accuracies) / len(accuracies)
    average_accuracies.append(average_accuracy)

model_versions = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']

plt.figure(figsize=(6, 4))
plt.plot(model_versions, average_accuracies, 'ro-')
plt.xlabel('Model Versions', fontsize=12)
plt.ylabel('Average Accuracy (%)', fontsize=12)
plt.title('Model Versions vs Average Accuracy', fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
