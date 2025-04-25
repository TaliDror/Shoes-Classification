
# Shoe Pair Classification CNN

This repository contains the PyTorch implementation of convolutional neural networks for classifying or matching left and right shoe pairs from images.

## Overview

The project includes two models:
- `CNN`: A convolutional neural network that processes a single RGB image.
- `CNNChannel`: A CNN that processes stacked RGB images of left and right shoes as a single 6-channel input.

The models are trained to predict whether a pair of shoes are a **match** or **mismatch**.

## File Structure

- `main.ipynb`: Main notebook for model training, evaluation, and experimentation.
- `models.py`: Contains the model definitions.
- `checkpoints/`: Directory for saved model weights (to be created).
- `data/`: Directory containing training and validation images (to be created).

## Installation

Clone the repository:
   ```bash
   git clone https://github.com/TaliDror/Shoes-Classification.git
   cd Shoes-Classification
```

## Dataset

To obtain the dataset and pretrained model checkpoints, download from google drive: 
data - https://drive.google.com/drive/u/1/folders/1WOUzqIfzjmZzLhLWYOUnhq9Dt2YgHNTv
checkpoints - https://drive.google.com/drive/u/1/folders/1xya8cXE15m7g8Ypck6-Cg4ZfjYTTp9Zm

After downloading:

Place the image data inside the data/ directory.

Place the .pth checkpoint files inside the checkpoints/ directory.

## Data Preprocessing

Before training, the data undergoes the following preprocessing steps:

 - Image Augmentation: Training images are augmented by flipping and rotating to increase data diversity.

 - Image Normalization: All images are normalized to a range of [-0.5, 0.5] for better model performance.

 - Data Splitting: The dataset is split into training, validation, and test sets, with separate subsets for different image types (e.g., woman and man).

 - Triplet Assignment: Images are organized into triplets based on unique IDs, with both original and augmented images assigned to their respective data arrays.

## Training
You can modify and run the training pipeline from main.ipynb. It includes training loops, evaluation, and model saving.

## Using The Model
Once trained, you can test the model using the provided example.
```python
# Generate positive and negative pairs
data_pos = generate_same_pair(test_w)
data_neg = generate_different_pair(test_w)

is_shown_correct = False
is_shown_incorrect = False

# Randomly select one correctly classified pair
rand_pos_idx = random.randint(0, len(data_pos) - 1)
xs_pos = torch.Tensor(data_pos[rand_pos_idx:rand_pos_idx+1]).transpose(1, 3).to(device)  # Assuming data_pos is correctly formatted
zs_pos = CNN_model(xs_pos)
pred_pos = zs_pos.max(1, keepdim=True)[1].detach().cpu().numpy()  # Get the predicted class index

# Display correctly classified positive example
plt.figure()
plt.imshow(data_pos[rand_pos_idx] + 0.5)  # Adjust brightness or normalization as needed
plt.title("Positive example: Correctly classified")

# Randomly select one incorrectly classified pair
rand_neg_idx = random.randint(0, len(data_neg) - 1)
xs_neg = torch.Tensor(data_neg[rand_neg_idx:rand_neg_idx+1]).transpose(1, 3).to(device)  # Assuming data_neg is correctly formatted
zs_neg = CNN_model(xs_neg)
pred_neg = zs_neg.max(1, keepdim=True)[1].detach().cpu().numpy()  # Get the predicted class index

# Display incorrectly classified negative example
plt.figure()
plt.imshow(data_neg[rand_neg_idx] + 0.5)  # Adjust brightness or normalization as needed
plt.title("Negative example: Incorrectly classified")

plt.show()
```
