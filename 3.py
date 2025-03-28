import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import logging

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_cnn_model(input_shape, num_classes):
    """
    Create a CNN model for ship classification.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of ship types.

    Returns:
        tensorflow.keras.Model: Compiled CNN model.
    """
    logger.info("Building CNN model for ship classification...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        BatchNormalization(),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info(f"CNN Model summary:\n{model.summary()}")
    return model


def fine_tune_model(base_model, num_classes):
    """
    Fine-tune a pre-trained model for automobile classification.

    Args:
        base_model (tensorflow.keras.Model): Pre-trained model.
        num_classes (int): Number of automobile classes.

    Returns:
        tensorflow.keras.Model: Fine-tuned model.
    """
    logger.info("Fine-tuning pre-trained model for automobile classification...")
    base_model.trainable = False  # Freeze the base model

    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info(f"Fine-tuned Model summary:\n{model.summary()}")
    return model


def train_model(model, train_generator, valid_generator, model_name, epochs=10):
    """
    Train a model with early stopping and model checkpointing.

    Args:
        model: Keras model.
        train_generator: Training data generator.
        valid_generator: Validation data generator.
        model_name (str): Name for saving the model.
        epochs (int): Maximum number of epochs.

    Returns:
        tuple: Trained model, training history.
    """
    # Create directory for model checkpoints if it doesn't exist
    checkpoint_dir = 'model_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, f'{model_name}_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # Train model
    logger.info(f"Training {model_name}...")
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    return model, history


def evaluate_model(model, test_generator):
    """
    Evaluate model performance.

    Args:
        model: Trained model.
        test_generator: Test data generator.

    Returns:
        float: Accuracy score.
    """
    logger.info("Evaluating model...")
    loss, accuracy = model.evaluate(test_generator, verbose=1)
    logger.info(f"Test accuracy: {accuracy:.4f}")
    return accuracy


def main():
    """
    Main function to orchestrate the entire process.
    """
    # Paths to datasets
    ships_dataset_path = 'datasets/ships_dataset'
    automobiles_dataset_path = 'datasets/cars_dataset'

    # Parameters
    input_shape = (128, 128, 3)
    batch_size = 32
    ship_classes = 10
    automobile_classes = 5

    # Data generators for ships dataset
    logger.info("Preparing data generators for ships dataset...")
    train_datagen = ImageDataGenerator(rescale=1.0/255)
    valid_datagen = ImageDataGenerator(rescale=1.0/255)
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(ships_dataset_path, 'train'),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )
    valid_generator = valid_datagen.flow_from_directory(
        os.path.join(ships_dataset_path, 'valid'),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )
    test_generator = test_datagen.flow_from_directory(
        os.path.join(ships_dataset_path, 'test'),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Step 1: Train CNN model for ship classification
    cnn_model = create_cnn_model(input_shape, ship_classes)
    cnn_model, cnn_history = train_model(
        cnn_model, train_generator, valid_generator, model_name='cnn_ships', epochs=10
    )
    ship_accuracy = evaluate_model(cnn_model, test_generator)

    # Step 2: Fine-tune for automobile classification
    logger.info("Preparing data generators for automobiles dataset...")
    train_generator_auto = train_datagen.flow_from_directory(
        os.path.join(automobiles_dataset_path, 'train'),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )
    valid_generator_auto = valid_datagen.flow_from_directory(
        os.path.join(automobiles_dataset_path, 'valid'),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )
    test_generator_auto = test_datagen.flow_from_directory(
        os.path.join(automobiles_dataset_path, 'test'),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Use MobileNetV2 as the base model for fine-tuning
    base_model = MobileNetV2(input_shape=input_shape,
                             include_top=False, weights='imagenet')
    fine_tuned_model = fine_tune_model(base_model, automobile_classes)
    fine_tuned_model, fine_tune_history = train_model(
        fine_tuned_model, train_generator_auto, valid_generator_auto, model_name='fine_tuned_automobiles', epochs=10
    )
    automobile_accuracy = evaluate_model(fine_tuned_model, test_generator_auto)

    # Log results
    logger.info(f"Ship classification accuracy: {ship_accuracy:.4f}")
    logger.info(
        f"Automobile classification accuracy: {automobile_accuracy:.4f}")


if __name__ == "__main__":
    main()

# Traceback (most recent call last):
#   File "F:\kod\llm-homework\hw1\3.py", line 246, in <module>
#     main()
#   File "F:\kod\llm-homework\hw1\3.py", line 217, in main
#     valid_generator_auto = valid_datagen.flow_from_directory(
#                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "F:\kod\llm-homework\hw1\venv\Lib\site-packages\keras\src\legacy\preprocessing\image.py", line 1138, in flow_from_directory
#     return DirectoryIterator(
#            ^^^^^^^^^^^^^^^^^^
#   File "F:\kod\llm-homework\hw1\venv\Lib\site-packages\keras\src\legacy\preprocessing\image.py", line 453, in __init__
#     for subdir in sorted(os.listdir(directory)):
#                          ^^^^^^^^^^^^^^^^^^^^^
# FileNotFoundError: [WinError 3] Sistem belirtilen yolu bulamıyor: 'datasets/cars_dataset\\valid'
