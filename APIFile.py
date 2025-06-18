from flask import Flask, request, jsonify, send_file, send_from_directory
import os
import tempfile
import shutil
from werkzeug.utils import secure_filename
import traceback
import cv2
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


# Import relevant functions from Predict.py
from Predict import load_model_weights, predict_single_image

app = Flask(__name__)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_request():
    """Log details about incoming requests"""
    logger.info(f"[{datetime.now()}] {request.method} {request.path} - Files: {list(request.files.keys())}")

# Load model at startup
MODEL_PATH = os.environ.get('MODEL_PATH', 'files/model.h5')
try:
    model = load_model_weights(MODEL_PATH)
    logger.info('Model loaded successfully.')
except Exception as e:
    logger.error(f'Failed to load model: {e}')
    model = None

@app.route('/ping', methods=['GET'])
def ping():
    log_request()
    return jsonify({'status': 'ok'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    log_request()
    
    if model is None:
        logger.error('Model not loaded')
        return jsonify({'error': 'Model not loaded.'}), 500
        
    if 'image' not in request.files:
        logger.error('No image file provided')
        return jsonify({'error': 'No image file provided.'}), 400
        
    image_file = request.files['image']
    if image_file.filename == '':
        logger.error('Empty filename provided')
        return jsonify({'error': 'Empty filename.'}), 400

    try:
        filename = secure_filename(image_file.filename)
        logger.info(f'Processing file: {filename}')
        
        # Create a temporary directory that won't be automatically deleted
        temp_dir = tempfile.mkdtemp()
        try:
            # Save uploaded file
            image_path = os.path.join(temp_dir, filename)
            image_file.save(image_path)
            
            # Process image
            mask, masked_img, rgba_img = predict_single_image(model, image_path, None, verbose=False)
            
            # Save outputs with unique filenames
            output_dir = os.path.join(temp_dir, 'outputs')
            os.makedirs(output_dir, exist_ok=True)
            
            mask_filename = 'mask.png'
            mask_path = os.path.join(output_dir, mask_filename)
            cv2.imwrite(mask_path, mask)
            
            # Clean up input file
            try:
                os.remove(image_path)
            except:
                pass
                
            logger.info(f'Successfully processed image, sending mask: {mask_path}')
            # Send file and ensure it's sent before we try to clean up
            response = send_from_directory(
                directory=output_dir,
                path=mask_filename,
                mimetype='image/png',
                as_attachment=True
            )
            
            # Set a callback to clean up the temp directory after the response is sent
            @response.call_on_close
            def cleanup():
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.info(f'Cleaned up temporary directory: {temp_dir}')
                except Exception as e:
                    logger.error(f'Error cleaning up temp directory: {e}')
            
            return response
            
        except Exception as e:
            # Clean up temp directory on error
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
            raise
            
    except Exception as e:
        logger.error(f'Error processing request: {str(e)}', exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
