# MonReader - Page Flip Detection Model

A deep learning solution for detecting page flips from single images, designed for the MonReader mobile document digitization system.

## Project Overview

MonReader is a mobile document digitization experience that automatically detects page flips from low-resolution camera previews. This project implements a CNN-based model to predict whether a page is being flipped using a single image.

### Problem Statement
- **Input**: Single images from video frames
- **Output**: Binary classification (flip vs not flip)
- **Goal**: High F1 score for accurate page flip detection

## Project Structure

```
Project 4 MonReader/
├── images/
│   ├── training/
│   │   ├── flip/          # Training images showing page flips
│   │   └── notflip/       # Training images without page flips
│   └── testing/
│       ├── flip/          # Testing images showing page flips
│       └── notflip/       # Testing images without page flips
├── data_loader.py         # Data loading and preprocessing
├── model.py              # CNN model architectures
├── evaluation.py         # Model evaluation and metrics
├── main.py              # Main training and evaluation pipeline
├── requirements.txt      # Python dependencies
└── README.md           # This file
```

## Installation

1. **Clone the repository** (if applicable)
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Explore the Dataset

First, explore your dataset to understand the data distribution:

```bash
python main.py --mode explore
```

This will show:
- Dataset statistics
- Class distribution
- Sample images from each class

### 2. Train a Model

Train a simple CNN model:

```bash
python main.py --mode train --model_type simple --epochs 50 --batch_size 32
```

Or train a transfer learning model:

```bash
python main.py --mode train --model_type transfer --epochs 30 --batch_size 16
```

**Parameters:**
- `--model_type`: `simple` (custom CNN) or `transfer` (pre-trained model)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate (default: 0.001)
- `--img_size`: Image dimensions (default: 224 224)
- `--save_plots`: Save evaluation plots

### 3. Evaluate a Trained Model

Evaluate a trained model on the test set:

```bash
python main.py --mode evaluate --model_path best_model.h5 --save_plots
```

### 4. Make Predictions on Single Images

Predict on a single image:

```bash
python main.py --mode predict --model_path best_model.h5 --image_path path/to/image.jpg
```

## Model Architectures

### 1. Simple CNN
- 4 convolutional blocks with batch normalization and dropout
- Dense layers with regularization
- Suitable for smaller datasets

### 2. Transfer Learning
- Uses pre-trained MobileNetV2 or ResNet50V2
- Fine-tuned for page flip detection
- Better performance with limited data

## Evaluation Metrics

The model is evaluated using:
- **F1 Score**: Primary metric (higher is better)
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **ROC AUC**: Area under ROC curve
- **PR AUC**: Area under Precision-Recall curve

## Key Features

### Data Loading (`data_loader.py`)
- Automatic image loading and preprocessing
- Train/validation split with stratification
- Image normalization and resizing
- Class distribution analysis
- Sample visualization

### Model Training (`model.py`)
- Custom CNN architecture
- Transfer learning with pre-trained models
- Early stopping and learning rate scheduling
- Model checkpointing
- Comprehensive evaluation metrics

### Evaluation (`evaluation.py`)
- Confusion matrix visualization
- ROC and Precision-Recall curves
- Training history plots
- Detailed classification reports
- Results saving and export

## Example Workflow

1. **Start with data exploration**:
   ```bash
   python main.py --mode explore
   ```

2. **Train a simple model**:
   ```bash
   python main.py --mode train --model_type simple --epochs 20
   ```

3. **Evaluate the model**:
   ```bash
   python main.py --mode evaluate --model_path best_model.h5 --save_plots
   ```

4. **Try transfer learning for better performance**:
   ```bash
   python main.py --mode train --model_type transfer --epochs 15
   ```

## Performance Tips

1. **Data Quality**: Ensure balanced class distribution
2. **Image Size**: 224x224 works well for most cases
3. **Transfer Learning**: Use for better performance with limited data
4. **Hyperparameter Tuning**: Experiment with learning rates and batch sizes
5. **Data Augmentation**: Consider adding rotation, zoom, and brightness variations

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or image size
2. **Poor Performance**: Try transfer learning or increase training epochs
3. **Overfitting**: Add more dropout or reduce model complexity
4. **Slow Training**: Use GPU acceleration if available

### Dependencies
- TensorFlow 2.15.0
- OpenCV 4.8.1
- NumPy 1.24.3
- Matplotlib 3.7.2
- Scikit-learn 1.3.0
- Pandas 2.0.3
- Pillow 10.0.0
- Seaborn 0.12.2

## Future Improvements

1. **Data Augmentation**: Add more augmentation techniques
2. **Ensemble Methods**: Combine multiple models
3. **Sequence Models**: Consider temporal information from video frames
4. **Real-time Processing**: Optimize for mobile deployment
5. **Multi-class Classification**: Detect different types of page movements

## License

This project is part of the MonReader document digitization system.

## Contact

For questions or issues, please refer to the project documentation or contact the development team. 