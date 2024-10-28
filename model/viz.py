from collections import Counter
import matplotlib.pyplot as plt
from .data import CLASS_ENCODER


def class_imblance(y):
    class_decoder = {v:k for k, v in CLASS_ENCODER.items()}
    y_counter = Counter(y)
    
    class_labels, counts = list(y_counter.keys()), list(y_counter.values())
    class_labels = [class_decoder[l] for l in class_labels]

    plt.figure(figsize=(10, 6))
    plt.bar(class_labels, counts, color='skyblue')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('Image Label Counts')
    plt.xticks(class_labels)
    plt.show()