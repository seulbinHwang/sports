from typing import Generator, Iterable, List, TypeVar
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm
from PIL import Image
from transformers import SiglipModel, SiglipProcessor
import cv2

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
    A classifier that uses a pre-trained SiglipModel for feature extraction.
    """

    def __init__(self, device: str = 'cuda', batch_size: int = 32):
        """
        Initialize the TeamClassifier with device and batch size.

        Args:
            device (str): The device to run the model on ('cpu' or 'cuda').
            batch_size (int): The batch size for processing images.
        """
        self.device = device
        self.batch_size = batch_size

        # Load the processor and model
        self.processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.features_model = SiglipModel.from_pretrained(
            SIGLIP_MODEL_PATH,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,  # torch_dtype 명시적으로 설정
            device_map=device
        )
        self.features_model.eval()  # Set model to evaluation mode

        # Prepare text inputs
        candidate_labels = ["red vest", "green vest", "white vest"]
        texts = [f'This is a photo of {label}.' for label in candidate_labels]

        # Compute text embeddings once and store them
        with torch.no_grad():
            # Tokenize text inputs
            text_inputs = self.processor(
                text=texts,
                images=None,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).to(self.device)

            # Compute text embeddings
            with torch.autocast(device_type=self.device, dtype=torch.float16):
                text_outputs = self.features_model.get_text_features(**text_inputs)
                # Normalize text embeddings
                self.text_embeddings = text_outputs / text_outputs.norm(dim=-1, keepdim=True)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops and predict labels.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted labels as a numpy array.
        """
        # Convert OpenCV images to PIL images
        crops_pil = [Image.fromarray(crop[:, :, ::-1]) for crop in crops]  # BGR에서 RGB로 변환
        batches = create_batches(crops_pil, self.batch_size)
        data = []

        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                # Process image inputs
                image_inputs = self.processor(
                    images=batch,
                    text=None,
                    return_tensors="pt",
                ).to(self.device)
                # 이미지 입력을 half precision으로 변환

                # Compute image embeddings
                with torch.autocast(device_type=self.device, dtype=torch.float16):  # 자동 혼합 정밀도 적용
                    image_outputs = self.features_model.get_image_features(**image_inputs)
                    # Normalize image embeddings
                    image_embeddings = image_outputs / image_outputs.norm(dim=-1, keepdim=True)

                # Compute similarity between image and text embeddings
                logits = image_embeddings @ self.text_embeddings.T
                probs = logits.softmax(dim=-1)
                predicted_ids = torch.argmax(probs, dim=1)
                data.append(predicted_ids.cpu().numpy())

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        pass  # Currently not implemented

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the labels for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted labels.
        """
        if len(crops) == 0:
            return np.array([])
        data = self.extract_features(crops)
        return data
