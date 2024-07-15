from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def train_model(model, train_generator, val_generator, epochs=10, output_path='output/best_model.keras'):
    # Ensure the output directory exists
    
    checkpoint = ModelCheckpoint(
        output_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        mode='min'
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        verbose=1,
        mode='min',
        min_lr=0.00001
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

    return history

