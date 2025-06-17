import os
import numpy as np
import cv2
import tensorflow as tf
from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path

# Hugging Face model configuration
HF_MODEL_ID = "Jenssss/basic_image_segmentatin"
MODEL_FILENAME = "model.h5"

from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    UpSampling2D,
    Concatenate,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
import argparse
from glob import glob
from tqdm import tqdm

# File paths
MODEL_PATH = "files/model.h5"
INPUT_DIR = "input/"
OUTPUT_DIR = "output/"
SINGLE_IMAGE_PATH = "test_image.jpg"
OUTPUT_MASK_PATH = "output_mask.png"

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

# Global image dimensions
image_h = 512
image_w = 512
MAX_WIDTH = 1024  # Maximum width for input images


def residual_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)

    s = Conv2D(num_filters, 1, padding="same")(inputs)
    s = BatchNormalization()(s)
    x = Activation("relu")(x + s)

    return x


def dilated_conv(inputs, num_filters):
    x1 = Conv2D(num_filters, 3, padding="same", dilation_rate=3)(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)

    x2 = Conv2D(num_filters, 3, padding="same", dilation_rate=6)(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    x3 = Conv2D(num_filters, 3, padding="same", dilation_rate=9)(inputs)
    x3 = BatchNormalization()(x3)
    x3 = Activation("relu")(x3)

    x = Concatenate()([x1, x2, x3])
    x = Conv2D(num_filters, 1, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def decoder_block(inputs, skip_features, num_filters):
    x = UpSampling2D((2, 2), interpolation="bilinear")(inputs)
    x = Concatenate()([x, skip_features])
    x = residual_block(x, num_filters)
    return x


def build_model(input_shape):
    """Input"""
    inputs = Input(input_shape)

    """ Pre-trained ResNet50 Model """
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = resnet50.get_layer("input_layer").output
    s2 = resnet50.get_layer("conv1_relu").output
    s3 = resnet50.get_layer("conv2_block3_out").output
    s4 = resnet50.get_layer("conv3_block4_out").output
    s5 = resnet50.get_layer("conv4_block6_out").output

    """ Bridge """
    b1 = dilated_conv(s5, 1024)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    y1 = UpSampling2D((8, 8), interpolation="bilinear")(d1)
    y1 = Conv2D(1, 1, padding="same", activation="sigmoid")(y1)

    y2 = UpSampling2D((4, 4), interpolation="bilinear")(d2)
    y2 = Conv2D(1, 1, padding="same", activation="sigmoid")(y2)

    y3 = UpSampling2D((2, 2), interpolation="bilinear")(d3)
    y3 = Conv2D(1, 1, padding="same", activation="sigmoid")(y3)

    y4 = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    outputs = Concatenate()([y1, y2, y3, y4])

    model = Model(inputs, outputs, name="U-Net")
    return model


def resize_with_max_width(image, max_width=MAX_WIDTH):
    """Resize image to have maximum width while maintaining aspect ratio"""
    h, w = image.shape[:2]

    if w <= max_width:
        # Image is already within the max width limit
        return image, (h, w)

    # Calculate new dimensions maintaining aspect ratio
    aspect_ratio = h / w
    new_width = max_width
    new_height = int(new_width * aspect_ratio)

    # Resize image
    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )

    return resized_image, (h, w)


def preprocess_image(image_path):
    """Preprocess input image for prediction"""
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Store original shape before any resizing
    original_shape = image.shape[:2]

    # First resize to max width if needed (maintaining aspect ratio)
    image, _ = resize_with_max_width(image, MAX_WIDTH)
    resized_shape = image.shape[:2]

    # Then resize to model input size
    image = cv2.resize(image, (image_w, image_h))
    image = image / 255.0
    image = image.astype(np.float32)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image, original_shape, resized_shape


def postprocess_mask(prediction, original_shape, resized_shape):
    """Postprocess prediction to create final mask"""
    # Take the average of all 4 output channels
    mask = np.mean(prediction[0], axis=-1)

    # Threshold the mask
    mask = (mask > 0.5).astype(np.uint8) * 255

    # First resize to the intermediate resized dimensions
    mask = cv2.resize(mask, (resized_shape[1], resized_shape[0]))

    # Then resize back to original dimensions
    mask = cv2.resize(mask, (original_shape[1], original_shape[0]))

    return mask


def apply_mask_to_image(image_path, mask, output_path=None):
    """Apply the segmentation mask to the original image"""
    # Read original image
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert mask to 3-channel for multiplication
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0

    # Apply mask to image
    masked_image = (original_image * mask_3ch).astype(np.uint8)

    # Create transparent background version (RGBA)
    mask_alpha = mask / 255.0
    rgba_image = np.zeros(
        (original_image.shape[0], original_image.shape[1], 4), dtype=np.uint8
    )
    rgba_image[:, :, :3] = original_image
    rgba_image[:, :, 3] = (mask_alpha * 255).astype(np.uint8)

    return masked_image, rgba_image


def predict_single_image(model, image_path, output_path=None, verbose=True):
    """Predict segmentation mask for a single image"""
    if verbose:
        print(f"Processing: {image_path}")

    # Create progress bar for single image processing steps
    if verbose:
        pbar = tqdm(total=6, desc="Processing steps", unit="step")

    # Preprocess image
    processed_image, original_shape, resized_shape = preprocess_image(image_path)
    if verbose:
        pbar.set_description("Preprocessing")
        pbar.update(1)
        print(f"\nOriginal size: {original_shape[1]}x{original_shape[0]}")
        if resized_shape != original_shape:
            print(
                f"\nResized to: {resized_shape[1]}x{resized_shape[0]} (max width: {MAX_WIDTH}px)"
            )

    # Predict
    if verbose:
        pbar.set_description("Predicting")
    prediction = model.predict(processed_image, verbose=0)
    if verbose:
        pbar.update(1)

    # Postprocess
    if verbose:
        pbar.set_description("Postprocessing")
    mask = postprocess_mask(prediction, original_shape, resized_shape)
    if verbose:
        pbar.update(1)

    # Apply mask to original image
    if verbose:
        pbar.set_description("Applying mask")
    masked_image, rgba_image = apply_mask_to_image(image_path, mask)
    if verbose:
        pbar.update(1)

    # Generate output paths
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = f"{base_name}_mask.png"
        masked_path = f"{base_name}_masked.png"
        rgba_path = f"{base_name}_transparent.png"
    else:
        base_name = os.path.splitext(output_path)[0]
        mask_path = f"{base_name}_mask.png"
        masked_path = f"{base_name}_masked.png"
        rgba_path = f"{base_name}_transparent.png"

    # Save results
    if verbose:
        pbar.set_description("Saving mask")
    cv2.imwrite(mask_path, mask)
    if verbose:
        pbar.update(1)

    if verbose:
        pbar.set_description("Saving results")
    cv2.imwrite(masked_path, masked_image)
    cv2.imwrite(rgba_path, rgba_image)
    if verbose:
        pbar.update(1)
        pbar.close()
        print(f"Results saved:")
        print(f"  - Mask: {mask_path}")
        print(f"  - Masked image: {masked_path}")
        print(f"  - Transparent PNG: {rgba_path}")

    return mask, masked_image, rgba_image


def predict_batch(model, input_dir, output_dir):
    """Predict segmentation masks for all images in a directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all image files
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(input_dir, ext)))
        image_files.extend(glob(os.path.join(input_dir, ext.upper())))

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process")

    # Process images with progress bar
    for image_path in tqdm(image_files, desc="Processing images", unit="img"):
        try:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}.png")
            predict_single_image(model, image_path, output_path, verbose=False)
        except Exception as e:
            tqdm.write(f"Error processing {image_path}: {str(e)}")

    print(f"\nCompleted! All results saved to: {output_dir}")
    print("Each image generates 3 files:")
    print("  - *_mask.png (binary mask)")
    print("  - *_masked.png (original image with background removed)")
    print("  - *_transparent.png (transparent PNG with alpha channel)")


def download_from_huggingface(model_id, filename, local_path):
    """Download a model file from Hugging Face Hub"""
    # Create directory if it doesn't exist
    os.makedirs(
        os.path.dirname(local_path) if os.path.dirname(local_path) else ".",
        exist_ok=True,
    )

    print(f"Downloading model from Hugging Face: {model_id}")
    try:
        # Download the specific file
        model_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=os.path.dirname(local_path),
            local_files_only=False,
            force_download=True,
            resume_download=True,
        )
        print("Model downloaded successfully!")
        return model_path
    except Exception as e:
        raise Exception(f"Failed to download model from Hugging Face: {str(e)}")


def load_model_weights(model_path, hf_model_id=None, hf_filename=None):
    """
    Load the trained model, downloading from Hugging Face Hub if not found locally

    Args:
        model_path (str): Local path to save/load the model
        hf_model_id (str, optional): Hugging Face model ID (e.g., 'username/model-name')
        hf_filename (str, optional): Filename of the model in the Hugging Face repository
    """
    # If model doesn't exist locally and Hugging Face info is provided, download it
    if not os.path.exists(model_path):
        if hf_model_id and hf_filename:
            print(f"Model not found at {model_path}")
            try:
                download_from_huggingface(hf_model_id, hf_filename, model_path)
            except Exception as e:
                raise Exception(f"Failed to download model: {str(e)}")
        else:
            raise FileNotFoundError(
                f"Model not found at {model_path} and no Hugging Face model ID provided. "
                "Please provide either a valid model path or Hugging Face model information."
            )

    print(f"Loading model from: {model_path}")

    # Build model architecture
    input_shape = (image_h, image_w, 3)
    model = build_model(input_shape)

    # Compile model (required for loading weights)
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(1e-4))

    # Load weights
    model.load_weights(model_path)
    print("Model loaded successfully!")

    return model


def main():
    global MAX_WIDTH

    parser = argparse.ArgumentParser(description="Person Segmentation Prediction")
    parser.add_argument(
        "input",
        nargs="?",
        default=SINGLE_IMAGE_PATH,
        help=f"Path to input image or directory [default: {SINGLE_IMAGE_PATH}]",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=MODEL_PATH,
        help=f"Path to the trained model file (.h5) [default: {MODEL_PATH}]",
    )
    parser.add_argument("--output", "-o", help="Path to output file or directory")
    parser.add_argument(
        "--batch", action="store_true", help="Process all images in input directory"
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=MAX_WIDTH,
        help=f"Maximum width for input images in pixels [default: {MAX_WIDTH}]",
    )

    args = parser.parse_args()

    # Update max width if specified
    MAX_WIDTH = args.max_width

    # Load model, will download if not found locally and URL is provided
    try:
        model = load_model_weights(
            args.model, hf_model_id=HF_MODEL_ID, hf_filename=MODEL_FILENAME
        )
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    print(f"Maximum input width set to: {MAX_WIDTH}px")

    # Process images
    if args.batch:
        if not os.path.isdir(args.input):
            print(f"Error: Input directory not found at {args.input}")
            return

        output_dir = args.output or "predictions"
        predict_batch(model, args.input, output_dir)
    else:
        if not os.path.isfile(args.input):
            print(f"Error: Input image not found at {args.input}")
            return

        try:
            predict_single_image(model, args.input, args.output)
        except Exception as e:
            print(f"Error processing image: {str(e)}")


if __name__ == "__main__":
    main()
