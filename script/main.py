import os
from tensorflow.keras.callbacks import ModelCheckpoint

from src.data_loader import ImageDataGenerator
def main():
    project_root = os.path.dirname(os.path.abspath("."))
    train_path = os.path.join(project_root, "train")
    val_path = os.path.join(project_root, "val")
    test_path = os.path.join(project_root,"test")
    output_path = os.path.join(project_root, "output", "CNN_model.h5")

    BATCH_SIZE = 32
    INPUT_SHAPE = (224, 224, 3)

    train_generrator,val_generator, test_generator = ImageDataGenerator(train_path, val_path, test_path)
    checkpoint = ModelCheckpoint(
        output_path,
        monitor = 'val_loss',
        verbose = 1,
        save_best_only = True,
        model = 'min'
    )
    history = model.fit(
        train_genrator,
        steps_per_epoch = 
    )

    print(train_path)

if __name__ == '__main__':
    main()