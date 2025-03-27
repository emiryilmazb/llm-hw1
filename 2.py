import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout,
    SpatialDropout1D, LSTM, Bidirectional, Concatenate, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os
import logging
from tqdm import tqdm
import random

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {e}")

# Load and explore the dataset
def load_data(file_path):
    """
    Load and preprocess the YouTube comments dataset
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Preprocessed dataset
    """
    try:
        logger.info(f"Loading dataset from {file_path}")
        df = pd.read_csv(file_path)
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Dataset contains missing values: {missing_values}")
            df = df.dropna()
            logger.info(f"Dropped rows with missing values. New shape: {df.shape}")
        
        # Verify sentiment labels
        unique_sentiments = df['Sentiment'].unique()
        logger.info(f"Unique sentiment labels: {unique_sentiments}")
        
        # Display class distribution
        sentiment_counts = df['Sentiment'].value_counts()
        logger.info(f"Class distribution:\n{sentiment_counts}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_text(text):
    """
    Preprocess text by removing special characters, numbers,
    converting to lowercase, and lemmatizing
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and lemmatize
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error in text preprocessing: {str(e)}")
        # Return original text if preprocessing fails
        return text

def prepare_data(df):
    """
    Prepare data for training: preprocess text, encode labels,
    and split into train and test sets
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, tokenizer, label_encoder
    """
    # Preprocess comments
    logger.info("Preprocessing text data...")
    df['Processed_Comment'] = df['Comment'].apply(preprocess_text)
    
    # Encode sentiment labels
    logger.info("Encoding sentiment labels...")
    label_encoder = LabelEncoder()
    df['Encoded_Sentiment'] = label_encoder.fit_transform(df['Sentiment'])
    
    # Split data into train and test sets
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['Processed_Comment'],
        df['Encoded_Sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=df['Encoded_Sentiment']
    )
    
    # Tokenize text
    logger.info("Tokenizing text...")
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    # Convert text to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences
    max_length = 100
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')
    
    return X_train_pad, X_test_pad, y_train, y_test, tokenizer, label_encoder, max_length

def build_cnn_model(vocab_size, embedding_dim, max_length, num_classes):
    """
    Build a CNN model for sentiment classification
    
    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of embeddings
        max_length (int): Maximum sequence length
        num_classes (int): Number of classes
        
    Returns:
        tensorflow.keras.Model: Compiled CNN model
    """
    logger.info("Building CNN model...")
    
    model = Sequential([
        # Embedding layer
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        SpatialDropout1D(0.2),
        
        # First convolutional block
        Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        GlobalMaxPooling1D(),
        
        # Dense layers
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"CNN Model summary:\n{model.summary()}")
    return model

def build_embedding_model(vocab_size, embedding_dim, max_length, num_classes):
    """
    Build an embedding-based model with bidirectional LSTM for sentiment classification
    
    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of embeddings
        max_length (int): Maximum sequence length
        num_classes (int): Number of classes
        
    Returns:
        tensorflow.keras.Model: Compiled embedding-based model
    """
    logger.info("Building embedding-based model with BiLSTM...")
    
    # Input layer
    input_layer = Input(shape=(max_length,))
    
    # Embedding layer
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(input_layer)
    embedding_dropout = SpatialDropout1D(0.2)(embedding_layer)
    
    # Bidirectional LSTM layer
    bilstm = Bidirectional(LSTM(64, return_sequences=True))(embedding_dropout)
    
    # Parallel convolutional layers with different kernel sizes
    conv1 = Conv1D(filters=64, kernel_size=3, padding='valid', activation='relu')(bilstm)
    conv2 = Conv1D(filters=64, kernel_size=4, padding='valid', activation='relu')(bilstm)
    conv3 = Conv1D(filters=64, kernel_size=5, padding='valid', activation='relu')(bilstm)
    
    # Global max pooling
    pool1 = GlobalMaxPooling1D()(conv1)
    pool2 = GlobalMaxPooling1D()(conv2)
    pool3 = GlobalMaxPooling1D()(conv3)
    
    # Concatenate pooled features
    concat = Concatenate()([pool1, pool2, pool3])
    
    # Dense layers
    dense1 = Dense(128, activation='relu')(concat)
    bn1 = BatchNormalization()(dense1)
    dropout1 = Dropout(0.3)(bn1)
    
    dense2 = Dense(64, activation='relu')(dropout1)
    bn2 = BatchNormalization()(dense2)
    dropout2 = Dropout(0.3)(bn2)
    
    # Output layer
    output_layer = Dense(num_classes, activation='softmax')(dropout2)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Embedding Model summary:\n{model.summary()}")
    return model

def train_model(model, X_train, y_train, X_test, y_test, model_name, batch_size=32, epochs=20):
    """
    Train a model with early stopping and model checkpointing
    
    Args:
        model: Keras model
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        model_name (str): Name for saving the model
        batch_size (int): Batch size
        epochs (int): Maximum number of epochs
        
    Returns:
        tuple: Trained model, training history
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
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        label_encoder: Label encoder
        
    Returns:
        float: Accuracy score
    """
    # Predict
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {accuracy:.4f}")
    
    # Classification report
    class_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=class_names)
    logger.info(f"Classification report:\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return accuracy

def plot_training_history(history, model_name):
    """
    Plot training history
    
    Args:
        history: Training history
        model_name (str): Name of the model
    """
    plt.figure(figsize=(12, 5))
    
    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    plt.close()

def predict_sentiment(model, tokenizer, label_encoder, text, max_length):
    """
    Predict sentiment for a given text
    
    Args:
        model: Trained model
        tokenizer: Fitted tokenizer
        label_encoder: Label encoder
        text (str): Input text
        max_length (int): Maximum sequence length
        
    Returns:
        str: Predicted sentiment
    """
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(padded_sequence)[0]
    predicted_class = np.argmax(prediction)
    sentiment = label_encoder.inverse_transform([predicted_class])[0]
    confidence = prediction[predicted_class]
    
    return sentiment, confidence

def main():
    """
    Main function to orchestrate the entire process
    """
    # Load data
    file_path = 'datasets/YoutubeCommentsDataSet.csv'
    df = load_data(file_path)
    
    # Prepare data
    X_train, X_test, y_train, y_test, tokenizer, label_encoder, max_length = prepare_data(df)
    
    # Model parameters
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 200
    num_classes = len(label_encoder.classes_)
    
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Number of classes: {num_classes}")
    
    # I. Build and train CNN model
    cnn_model = build_cnn_model(vocab_size, embedding_dim, max_length, num_classes)
    cnn_model, cnn_history = train_model(
        cnn_model, X_train, y_train, X_test, y_test, 
        model_name='cnn_sentiment'
    )
    
    # Evaluate CNN model
    logger.info("Evaluating CNN model...")
    cnn_accuracy = evaluate_model(cnn_model, X_test, y_test, label_encoder)
    plot_training_history(cnn_history, 'CNN')
    
    # II. Build and train embedding-based model
    embedding_model = build_embedding_model(vocab_size, embedding_dim, max_length, num_classes)
    embedding_model, embedding_history = train_model(
        embedding_model, X_train, y_train, X_test, y_test, 
        model_name='embedding_sentiment'
    )
    
    # Evaluate embedding model
    logger.info("Evaluating embedding-based model...")
    embedding_accuracy = evaluate_model(embedding_model, X_test, y_test, label_encoder)
    plot_training_history(embedding_history, 'Embedding')
    
    # III. Compare model performances
    logger.info("Model performance comparison:")
    logger.info(f"CNN model accuracy: {cnn_accuracy:.4f}")
    logger.info(f"Embedding model accuracy: {embedding_accuracy:.4f}")
    
    # Test with sample comments
    sample_comments = [
        "This video is amazing! I learned so much from it!",
        "The content is okay, but the presentation could be better.",
        "Worst tutorial ever. Complete waste of time."
    ]
    
    logger.info("Testing with sample comments:")
    for comment in sample_comments:
        # Predict using the better performing model
        if embedding_accuracy > cnn_accuracy:
            sentiment, confidence = predict_sentiment(
                embedding_model, tokenizer, label_encoder, comment, max_length
            )
            model_name = "Embedding model"
        else:
            sentiment, confidence = predict_sentiment(
                cnn_model, tokenizer, label_encoder, comment, max_length
            )
            model_name = "CNN model"
        
        logger.info(f"Comment: '{comment}'")
        logger.info(f"Predicted sentiment ({model_name}): {sentiment} (confidence: {confidence:.4f})")
        logger.info("-" * 50)
    
    # Save the better performing model
    if embedding_accuracy > cnn_accuracy:
        embedding_model.save('best_sentiment_model.h5')
        logger.info("Saved embedding model as the best performer.")
    else:
        cnn_model.save('best_sentiment_model.h5')
        logger.info("Saved CNN model as the best performer.")

if __name__ == "__main__":
    main()