
from tensorflow.keras.preprocessing import ImageDataGenerator

def data_generator(train_path, val_path, test_path, img_size=(224, 224), train_batch_size=32, val_batch_size = 16):
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest',
    )

    val_datagen = ImageDataGenerator(rescale= 1./ 255)
    test_datagen = ImageDataGenerator(rescale= 1./255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        batch_size=train_batch_size,
        class_mode='binary',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=img_size,
        batch_size=val_batch_size,
        class_mode='binary',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=img_size,
        batch_size=train_batch_size,
        class_mode='binary'
    )

    return train_generator, val_generator, test_generator