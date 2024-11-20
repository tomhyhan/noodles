import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from .data import CLASS_ENCODER


def class_imbalance(y):
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


def draw_loss(training_loss, window=5):
    training_loss = np.array(training_loss)
    
    kernel = np.ones(window) / window
    moving_avg = np.convolve(training_loss, kernel, mode="valid")
    
    plt.figure(figsize=(8, 6))
    plt.plot(training_loss, label="Loss history", marker='o', color="blue")
    plt.plot(moving_avg, color="red")
    
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    
    plt.title("Loss over Iterations")

    plt.show()
    
def draw_train_val_accuracy(training_acc, val_acc):
    plt.figure(figsize=(8, 6))
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.plot(training_acc, label="Training Accuracy")
    plt.plot(val_acc, color="orange", label="Val Accuracy")
    
    
    plt.show()

def show_image(image):
    pass
