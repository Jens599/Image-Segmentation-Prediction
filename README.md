# Image Segmentation Project

This project implements a deep learning-based image segmentation system using TensorFlow and a custom U-Net like architecture. The model is trained to perform semantic segmentation on images.

## Features

- **Automatic Model Download**: Automatically downloads the pre-trained model from Hugging Face Hub if not found locally
- **Flexible Input**: Process single images or batch process entire directories
- **Multiple Output Formats**: Generates various output formats including masks and overlays
- **Configurable**: Adjustable parameters for different use cases

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/image-segmentation.git
   cd image-segmentation
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install TensorFlow with GPU support**
   
   For GPU acceleration, install TensorFlow with GPU support:
   
   ```bash
   # For CUDA 11.8 and cuDNN 8.6 (recommended for most NVIDIA GPUs)
   pip install tensorflow[and-cuda]
   
   # Or for specific versions
   # pip install tensorflow==2.12.0 --upgrade
   ```
   
   > **Note**: Ensure you have the appropriate [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx) and [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) installed.

4. **Install remaining dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify GPU Setup**
   Run the following command to verify TensorFlow can access your GPU:
   ```bash
   python -c "import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"
   ```
   You should see `Num GPUs Available: 1` or more if your GPU is properly configured.

## Usage

### Basic Usage

```bash
python Predict.py path/to/your/image.jpg --output output_directory/
```

### Batch Processing

To process all images in a directory:

```bash
python Predict.py input_directory/ --batch --output output_directory/
```

### Command Line Arguments

- `input`: Path to input image or directory (default: "input/")
- `--model`, `-m`: Path to the trained model file (default: "files/model.h5")
- `--output`, `-o`: Path to output directory (default: "output/")
- `--batch`: Process all images in input directory
- `--max-width`: Maximum width for input images in pixels (default: 1024)

## Project Structure

```
.
├── input/                  # Input images
├── output/                 # Output directory for predictions
├── files/                  # Model files (automatically created)
│   └── models.md          # Model information
├── Predict.py              # Main prediction script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Model Details

The model is based on a U-Net like architecture with a ResNet50 encoder. It's trained to perform semantic segmentation on various types of images.

### Model Architecture

- **Encoder**: ResNet50 (pre-trained on ImageNet)
- **Decoder**: Custom U-Net like architecture with skip connections
- **Input Shape**: 512x512x3 (RGB)
- **Output Shape**: 512x512x1 (segmentation mask)

## Output Formats

For each input image, the following outputs are generated:

1. `{filename}_mask.png`: Binary segmentation mask
2. `{filename}_overlay.png`: Original image with segmentation mask overlay
3. `{filename}_transparent.png`: Transparent PNG with segmented area

## Troubleshooting

1. **CUDA Out of Memory**
   - Reduce the input image size using `--max-width`
   - Close other GPU-intensive applications

2. **Model Download Issues**
   - Check your internet connection
   - Verify the Hugging Face model ID in `Predict.py`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Built with TensorFlow and Keras
- Model hosted on Hugging Face Hub
- Uses OpenCV for image processing
