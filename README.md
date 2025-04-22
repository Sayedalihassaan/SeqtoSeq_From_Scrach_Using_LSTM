## Sentiment Analysis with RNN and LSTM on IMDB

### Dataset
This repository contains a Jupyter notebook implementing sentiment analysis on the IMDB movie review dataset using Recurrent Neural Networks (RNNs) and Bidirectional Long Short-Term Memory (LSTM) models. The project classifies movie reviews as positive or negative, demonstrating a complete natural language processing (NLP) pipeline, including data preprocessing, model training, evaluation, and inference using TensorFlow.
Table of Contents

Project Overview
Features
Installation
Usage
Project Structure
Dataset
Methodology
Results
Future Improvements
Contributing
License
Acknowledgements

Project Overview
The goal of this project is to build and compare two deep learning models for binary sentiment classification:

A Simple RNN model with stacked layers.
A Bidirectional LSTM model for improved context understanding.

Both models are trained on the IMDB dataset to predict sentiment in movie reviews. The project showcases key NLP techniques, such as word embeddings, sequence padding, and GPU-accelerated training, while providing a foundation for experimenting with recurrent architectures.
Features

Data Preprocessing: Tokenizes reviews, pads sequences, and maps indices to words for interpretability.
Model Architectures:
Simple RNN with two layers (64 and 32 units).
Bidirectional LSTM with two layers (64 and 32 units) for bidirectional context.


Training: Uses early stopping to optimize training and prevent overfitting.
Evaluation: Visualizes training/validation accuracy and loss curves; computes test set performance.
Inference: Includes a function to classify new reviews as positive or negative.
Model Persistence: Saves and loads the trained model for reuse.
GPU Support: Leverages TensorFlow for GPU-accelerated training.

Installation
To run this project locally, follow these steps:

Clone the Repository:
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:Install the required libraries listed in requirements.txt:
pip install -r requirements.txt

The requirements.txt includes:
tensorflow==2.17.0
numpy==2.1.1
pandas==2.2.3
matplotlib==3.9.2
seaborn==0.13.2


Optional: GPU Support:For GPU acceleration, ensure you have a CUDA-compatible GPU and install TensorFlow with GPU support:
pip install tensorflow[and-cuda]

Verify GPU availability:
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))



Usage

Run the Notebook:Open the RNN.ipynb notebook in Jupyter Notebook or Google Colab:
jupyter notebook RNN.ipynb

Execute the cells to preprocess data, train models, evaluate performance, and perform inference.

Inference with the Trained Model:Use the inferance function to classify a new review. Example:
from tensorflow.keras.models import load_model

model = load_model("model_saveed.keras")
text = "This movie was absolutely fantastic and thrilling!"
prediction = inferance(text, model)
print(f"Sentiment: {prediction}")

The function returns "Positive" or "Negative" based on the model's prediction.

Visualize Results:The notebook generates plots for training and validation accuracy/loss, viewable in the notebook or exportable as images.


Project Structure
your-repo-name/
├── RNN.ipynb                   # Main notebook with project code
├── model_saveed.keras         # Saved Bidirectional LSTM model
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── LICENSE                    # License file (e.g., MIT)

Dataset
The IMDB dataset is sourced from TensorFlow's tf.keras.datasets.imdb. Key details:

Training Set: 25,000 movie reviews.
Test Set: 25,000 movie reviews.
Labels: Binary (0 = negative, 1 = positive).
Vocabulary: Top 10,000 most frequent words.
Sequence Length: Padded/truncated to 300 tokens.

Methodology
The project follows a structured NLP pipeline:

Data Loading:
Loads the IMDB dataset with a vocabulary size of 10,000.
Creates word-to-index and index-to-word mappings.


Preprocessing:
Pads sequences to 300 tokens using post-padding and truncation.
Reconstructs text from indices for interpretability.


Model Architectures:
Simple RNN:
Embedding layer (10,000 vocab size, 128 dimensions).
Two RNN layers (64 units with return_sequences=True, 32 units).
Dense layers (16 units with ReLU, 1 unit with sigmoid).


Bidirectional LSTM:
Embedding layer (same as above).
Two bidirectional LSTM layers (64 units with return_sequences=True, 32 units).
Dense layers (same as above).




Training:
Binary cross-entropy loss, Adam optimizer (learning rate 0.01).
Batch size: 32, max epochs: 50, early stopping (patience=3) based on validation loss.


Evaluation:
Plots accuracy and loss curves for training and validation.
Computes test set accuracy and loss.


Inference:
Preprocesses input text, pads sequences, and predicts sentiment using the trained model.



Tools and Libraries



Library
Purpose



tensorflow
Model building, training, inference


numpy
Numerical operations


pandas
Data manipulation (included, unused)


matplotlib
Plotting accuracy/loss curves


seaborn
Enhanced visualizations (included, unused)


Results

Simple RNN:
Test Accuracy: ~50.03% (near random guessing, likely due to vanishing gradients).
Test Loss: ~0.6931.
Observation: Poor convergence, as seen in flat accuracy/loss curves.


Bidirectional LSTM:
Defined but not trained in the notebook. Expected to perform better due to handling long-term dependencies.
Inference example classifies a review as "Positive" (untrained model, unreliable).


Visualizations:
Accuracy/loss plots highlight the Simple RNN's inability to learn effectively.
Plots are generated using Matplotlib and displayed in the notebook.



Note: The Simple RNN's poor performance suggests limitations with the IMDB dataset's long sequences. Training the Bidirectional LSTM model is recommended for improved results.
Future Improvements

Train the LSTM Model: Complete training and evaluation of the Bidirectional LSTM.
Hyperparameter Tuning: Optimize learning rate, batch size, and layer sizes.
Advanced Architectures: Experiment with GRU, Transformer-based models (e.g., BERT), or attention mechanisms.
Preprocessing Enhancements: Apply stemming, lemmatization, or stop-word removal.
Regularization: Add dropout or L2 regularization to prevent overfitting.
Error Analysis: Analyze misclassified reviews to improve model robustness.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make your changes and commit (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Please ensure your code follows PEP 8 style guidelines and includes relevant documentation.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgements

TensorFlow/Keras: For the IMDB dataset and deep learning framework.
IMDB Dataset: For the publicly available movie review data.
Matplotlib: For visualization tools.


Feel free to star ⭐ this repository if you find it useful! For questions or feedback, open an issue or contact me at [saiedhassaan2@gmail.com].
