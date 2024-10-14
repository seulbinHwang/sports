from typing import Generator, Iterable, List, TypeVar

import numpy as np
import supervision as sv
import torch
from tqdm import tqdm
import PIL
from transformers import SiglipModel, SiglipProcessor

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        """
       Initialize the TeamClassifier with device and batch size.

       Args:
           device (str): The device to run the model on ('cpu' or 'cuda').
           batch_size (int): The batch size for processing images.
       """
        self.device = device
        self.batch_size = batch_size
        self.processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.features_model = SiglipModel.from_pretrained(
    SIGLIP_MODEL_PATH,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,  # 반 정밀도 사용
    device_map=device,
)
        self.features_model.eval()  # Set model to evaluation mode


    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        candidate_labels = ["red team", "white team"]
        texts = [f'This is a photo of {label}.' for label in candidate_labels]
        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                batch: List[PIL.Image.Image] # len(batch) = 10
                inputs = self.processor(
                    text=texts, padding="max_length",
                    images=batch, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    with torch.autocast(device_type=self.device, dtype=torch.float16):
                        outputs = self.features_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = torch.sigmoid(logits_per_image)  # these are the probabilities
                predicted_ids = torch.argmax(probs, dim=1)  # (n, 2) -> (n,)
                # embeddings: (10, 768)
                # embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(predicted_ids.cpu().numpy())

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        return

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the cluster labels for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if len(crops) == 0:
            return np.array([])
        data = self.extract_features(crops)
        return data
