# Image Captioning with CNN and LSTM

This project demonstrates an image captioning system that uses a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The model is built using TensorFlow and Keras, and it utilizes VGG16 for feature extraction from images and LSTM for generating descriptive captions.

## Project Structure
- `captions.txt`: File containing image IDs and their corresponding captions.
- `Images/`: Directory containing images for feature extraction and captioning.
- `features.pkl`: Pickled file containing extracted features from images.
- `best_model.h5`: Saved model after training.

## Dataset
The dataset used in this project is the [Dataset: Flickr30k](https://www.kaggle.com/datasets/adityajn105/flickr30k) dataset from Kaggle. It contains images and their corresponding captions.
Images: Stored in the Images/ directory.
Captions: Stored in captions.txt, with each line containing an image ID followed by its captions, separated by commas.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Pillow
- tqdm
- nltk
## How to Use
1.Feature Extraction: Run the script to extract features from images using VGG16 and save them in features.pkl.

2.Data Preprocessing: Preprocess the captions by cleaning and tokenizing them. This step also splits the dataset into training and testing sets.

3.Model Training: Train the image captioning model using the extracted features and preprocessed captions. The model architecture includes an encoder for image features and a decoder for generating captions.

4.Caption Prediction: Use the trained model to generate captions for new images. The script provides a function to load an image and predict its caption.

## Training
The model will be trained for a specified number of epochs and saved as best_model.h5.

## Evaluation
After training, the model is evaluated using BLEU scores to measure the quality of generated captions compared to the actual captions.

## Example
To generate a caption for a specific image, use the generate_caption function:
```python
generate_caption("1000092795.jpg")
```
This function will display the actual and predicted captions for the provided image.

## Note
Ensure that the images and captions are properly aligned and formatted.

## Snapshots


## Acknowledgement
-The VGG16 model is pre-trained on ImageNet and is available via TensorFlow and Keras.
-BLEU score implementation is provided by the nltk library.
-Dataset: [Dataset: Flickr30k](https://www.kaggle.com/datasets/adityajn105/flickr30k) from Kaggle.
-Adjust hyperparameters and model architecture as needed based on your specific use case and dataset.

