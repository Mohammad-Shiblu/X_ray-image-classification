from tensorflow.keras.models import load_model

def evaluate_model(model_path, test_generator):
    model = load_model(model_path)
    test_loss, test_acc = model.evaluate(
        test_generator,
        steps=test_generator.samples // test_generator.batch_size,
        verbose=1
    )
    return test_loss, test_acc
