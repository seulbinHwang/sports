from typing import Generator, Iterable, List, TypeVar
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm
from PIL import Image
import mobileclip

V = TypeVar("V")

# MobileCLIP 모델 경로
MOBILECLIP_MODEL_PATH = './mobileclip_s0.pt'  # 사전에 다운로드한 경로

def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
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
    MobileCLIP 모델을 사용하는 팀 분류기.
    """
    def __init__(self, device: str = 'cuda', batch_size: int = 256):
        """
        팀 분류기 초기화.

        Args:
            device (str): 모델을 실행할 장치 ('cpu' 또는 'cuda').
            batch_size (int): 이미지 처리 배치 크기.
        """
        self.device = device
        self.batch_size = batch_size

        # MobileCLIP 모델과 토크나이저 및 전처리기 불러오기
        self.model, _, self.preprocess = mobileclip.create_model_and_transforms(
            'mobileclip_s0', pretrained=MOBILECLIP_MODEL_PATH
        )
        self.tokenizer = mobileclip.get_tokenizer('mobileclip_s0')

        # 모델을 지정한 디바이스로 이동 및 평가 모드로 전환
        self.model.to(self.device)
        self.model.eval()
        self.model.half()  # 모델을 half precision으로 변환

        # 텍스트 특징을 한 번만 계산하여 저장
        candidate_labels = ["red vest", "green vest", "white vest"]
        texts = self.tokenizer([f'This is a photo of {label}.' for label in candidate_labels])

        with torch.no_grad():
            # 텍스트를 토큰화하고 GPU로 이동
            texts = texts.to(self.device)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                text_features = self.model.encode_text(texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            self.text_features = text_features.half()

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        주어진 이미지 크롭에서 MobileCLIP을 사용해 특징을 추출.

        Args:
            crops (List[np.ndarray]): 이미지 크롭 목록.

        Returns:
            np.ndarray: 추출된 특징들.
        """
        # 이미지를 한꺼번에 PIL 이미지로 변환
        crops_pil = [Image.fromarray(crop[..., ::-1]) for crop in crops]
        batches = create_batches(crops_pil, self.batch_size)
        data = []

        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                # 배치 단위로 전처리 및 GPU로 이동
                processed_images = torch.stack([self.preprocess(image) for image in batch]).to(self.device)
                processed_images = processed_images.half()  # float16 사용

                with torch.cuda.amp.autocast(dtype=torch.float16):
                    image_features = self.model.encode_image(processed_images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # 미리 계산한 텍스트 특징 사용
                probs = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                predicted_ids = torch.argmax(probs, dim=1)
                data.append(predicted_ids.cpu().numpy())

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        return

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        return data
