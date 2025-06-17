import tensorflow as tf
import os

# Suppress TensorFlow warnings for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Show GPU info but reduce other warnings

print("TensorFlow version:", tf.__version__)
print("CUDA available:", tf.test.is_built_with_cuda())

# Use newer GPU detection method
gpus = tf.config.list_physical_devices("GPU")
print("GPU available:", len(gpus) > 0)

# List all available devices
print("\nAvailable devices:")
for device in tf.config.list_physical_devices():
    print(f"  {device}")

# Specifically check for GPUs
gpus = tf.config.list_physical_devices("GPU")
print(f"\nGPU devices found: {len(gpus)}")
for i, gpu in enumerate(gpus):
    print(f"  GPU {i}: {gpu}")

# Test GPU computation
if gpus:
    print("\nTesting GPU computation...")
    with tf.device("/GPU:0"):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print("Matrix multiplication result:", c.numpy())
    print("GPU test successful!")
else:
    print("\nNo GPU found. Running on CPU.")
