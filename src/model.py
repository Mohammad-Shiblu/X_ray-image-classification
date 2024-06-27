
from tensorflow.keras.model import Sequential
from tesnorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_CNN(input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation = 'relu', input_shape = input_shape),
        MaxPooling2D((2, 2,)),
        Conv2D(64, (3,3), activation = 'relu'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3,), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(256, activation = 'relu'),
        Dropout(0.5),
        Dense(1, activation = 'sigmoid')
    ])

    model.compile(optimizer='adam', loss = 'binary_cross_entropy', metrics = ['accuracy'])
    return model