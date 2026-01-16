"""CNN model predictor for mushroom species identification."""

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from typing import Dict

from config import CNN_MODEL_PATH, IMAGE_SIZE, MUSHROOM_SPECIES


class MushroomPredictor:
    """Handles CNN-based mushroom species prediction."""

    def __init__(self, model_path: str = CNN_MODEL_PATH):
        """
        Initialize the predictor with a trained CNN model.

        Args:
            model_path: Path to the trained Keras model
        """
        self.model = load_model(model_path)
        self.species = MUSHROOM_SPECIES

    def predict(self, img_path: str, top_k: int = 3) -> Dict[str, float]:
        """
        Predict mushroom species from an image.

        Args:
            img_path: Path to the image file
            top_k: Number of top predictions to return

        Returns:
            Dictionary mapping species names to confidence scores
        """
        # Load and preprocess image
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Get predictions
        predictions = self.model.predict(img_array)
        predictions_flat = predictions.ravel()

        # Get top k predictions
        indexes = np.argpartition(predictions_flat, -top_k)[-top_k:]
        values = predictions_flat[indexes]

        # Sort in descending order
        sorted_idx = np.argsort(values)[::-1]
        indexes, values = indexes[sorted_idx], values[sorted_idx]
        species_names = [self.species[i] for i in indexes]

        return dict(zip(species_names, values))