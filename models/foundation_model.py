import torch
import open_clip
from torchvision import transforms
import torchvision.models as models


class ModelLoader:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.clip_model = None
        self.clip_preprocess = None

        self.dino_model = None
        self.dino_preprocess = None

        self.resnet_model = None
        self.resnet_preprocess = None

    # ===== CLIP =====
    def load_clip(self):
        if self.clip_model is None:  
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="openai"
            )
            model = model.to(self.device).eval()

            preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                )
            ])

            self.clip_model = model
            self.clip_preprocess = preprocess

        return self.clip_model, self.clip_preprocess

    # ===== DINO =====
    def load_dino(self):
        if self.dino_model is None:
            model = torch.hub.load(
                "facebookresearch/dino:main",
                "dino_vitb16"
            )
            model = model.to(self.device).eval()

            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ])

            self.dino_model = model
            self.dino_preprocess = preprocess

        return self.dino_model, self.dino_preprocess

    # ===== RESNET =====
    def load_resnet(self):
        if self.resnet_model is None:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            model = torch.nn.Sequential(*list(model.children())[:-1])
            model = model.to(self.device).eval()

            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ])

            self.resnet_model = model
            self.resnet_preprocess = preprocess

        return self.resnet_model, self.resnet_preprocess