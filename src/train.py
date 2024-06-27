from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(model, train_generator, val_generator, epochs= 10):
    
    checkpoint = ModelCheckpoint(
        'best_model.h5',
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

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping]
    )

    return history