import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import data_generator
from src.model import create_CNN, create_resnet50
from src.train import train_model
from src.test import evaluate_model


def main():
    train_dir = "data/train"
    val_dir = "data/val"
    test_dir = "data/test"
    output_dir = "output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    cnn_best_model_path = os.path.join(output_dir, "CNN_model.keras")
    resnet_best_model_path = os.path.join(output_dir, "resnet_best_model.keras")
    EPOCHS = 10

    # Load data
    train_generator, val_generator, test_generator = data_generator(train_dir, val_dir, test_dir, img_size=(224, 224))
    
    # Train CNN
    cnn_model = create_CNN(input_shape=(224, 224, 3))
    cnn_history = train_model(cnn_model, train_generator, val_generator, EPOCHS, cnn_best_model_path)
    print(f"CNN training complete. Best model saved as '{cnn_best_model_path}'.")
    cnn_test_loss, cnn_test_acc = evaluate_model(cnn_best_model_path, test_generator)
    print(f"CNN Test Loss: {cnn_test_loss:.4f}, Test Accuracy: {cnn_test_acc:.4f}")
    
    # Train ResNet-50
    resnet_model = create_resnet50(input_shape=(224, 224, 3), num_classes=1)
    resnet_history = train_model(resnet_model, train_generator, val_generator, EPOCHS, resnet_best_model_path)
    print(f"ResNet-50 training complete. Best model saved as '{resnet_best_model_path}'.")
    resnet_test_loss, resnet_test_acc = evaluate_model(resnet_best_model_path, test_generator)
    print(f"ResNet-50 Test Loss: {resnet_test_loss:.4f}, Test Accuracy: {resnet_test_acc:.4f}")
    
    # Compare results
    print("\nComparison of CNN and ResNet-50:")
    print(f"CNN Test Accuracy: {cnn_test_acc:.4f}")
    print(f"ResNet-50 Test Accuracy: {resnet_test_acc:.4f}")
    
    
if __name__ == '__main__':
    main()
