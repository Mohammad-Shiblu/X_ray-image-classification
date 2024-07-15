# X-ray Image Classification

This project aims to classify X-ray images using two different neural network architectures: a custom Convolutional Neural Network (CNN) and a pre-trained ResNet-50 model. The project compares the performance of these models on the given dataset.

## Project Structure
- `data/`: Contains the training, validation, and test datasets.
- `output/`: Directory where the best models will be saved.
- `script/main.py`: Main script to run the training and evaluation.
- `src/`: Contains the modules for data loading, model definition, and training.
- `test.py`: Script for evaluating the models.

## Setup and Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-username/X_ray-image-classification.git
    cd X_ray-image-classification
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Ensure the data directories**:
    Place your X-ray images into the `data/train`, `data/val`, and `data/test` directories, with subdirectories for each class.

## Running the Project

To train and evaluate the models, run the main script:

```sh
python script/main.py
