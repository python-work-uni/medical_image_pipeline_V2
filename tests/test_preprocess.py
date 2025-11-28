import pytest
import numpy as np
import sys
import os
from unittest.mock import patch

# -------------------------------------------------------------------------
# PATH SETUP (Tailored for your structure)
# Your structure is:
#   ROOT/
#     ├── src/
#     │    └── preprocess.py
#     └── tests/
#          └── test_preprocess.py
#
# We need to tell Python: "Start at this file, go up one level (to ROOT), 
# then go down into 'src' to find modules."
# -------------------------------------------------------------------------
current_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(current_dir, '../src'))
sys.path.append(src_path)

# Now we can safely import from your src folder
from preprocess import preprocess_image

# -------------------------------------------------------------------------
# TEST SUITE
# -------------------------------------------------------------------------

@patch('preprocess.cv2.imread')
def test_preprocess_image_shape(mock_imread):
    """
    Test that the function correctly resizes any input image to (224, 224).
    """
    # 1. Arrange: Create a fake random image (100x100, 3 channels)
    fake_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mock_imread.return_value = fake_image

    # 2. Act: Run the function
    result = preprocess_image("dummy/path/image.jpg")

    # 3. Assert: Verify the output shape is exactly (224, 224, 3)
    assert result.shape == (224, 224, 3)

@patch('preprocess.cv2.imread')
def test_preprocess_image_normalization(mock_imread):
    """
    Test that pixel values are normalized to the 0.0 - 1.0 range.
    """
    # 1. Arrange: Create a fake image with known max value
    fake_image = np.array([[[0, 128, 255]]], dtype=np.uint8)
    mock_imread.return_value = fake_image

    # 2. Act
    result = preprocess_image("dummy/path/image.jpg")

    # 3. Assert: Check float conversion and range
    assert result.max() <= 1.0
    assert result.min() >= 0.0
    assert result.dtype == 'float32'

@patch('preprocess.cv2.imread')
def test_preprocess_image_failure(mock_imread):
    """
    Test that the function handles missing images gracefully (returns None).
    """
    # 1. Arrange: Simulate cv2.imread failing (returning None)
    mock_imread.return_value = None

    # 2. Act
    result = preprocess_image("non_existent_file.jpg")

    # 3. Assert
    assert result is None