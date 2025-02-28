import os
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import time
from transformers import DistilBertModel , DistilBertConfig , DistilBertTokenizer
import segmentation_models_pytorch_3d as smp
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader ,random_split
import nibabel as nib
from tqdm import tqdm
from sklearn.metrics import precision_score,recall_score
import shutil
import torch.optim as optim
import shutil
import random
import scipy.ndimage
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nibabel as nib
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Window:
    def __init__(self, window_min, window_max):
        self.window_min = window_min
        self.window_max = window_max

    def __call__(self, image):
        return np.clip(image, self.window_min, self.window_max)

class MinMaxNorm:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, image):
        return (image - self.low) / (self.high - self.low)

class RandomTranslation:
    def __init__(self, max_shift=10):
        self.max_shift = max_shift

    def __call__(self, image):
        shifts = [random.randint(-self.max_shift, self.max_shift) for _ in range(3)]
        return scipy.ndimage.shift(image, shift=shifts, mode='nearest')

class RandomFlip:
    def __call__(self, image):
        if random.random() > 0.5:
            image = np.flip(image, axis=0)
        if random.random() > 0.5:
            image = np.flip(image, axis=1)
        if random.random() > 0.5:
            image = np.flip(image, axis=2)
        return image

class RandomRotation:
    def __init__(self, angle_range=10):
        self.angle_range = angle_range

    def __call__(self, image):
        angle_x = random.uniform(-self.angle_range, self.angle_range)
        angle_y = random.uniform(-self.angle_range, self.angle_range)
        angle_z = random.uniform(-self.angle_range, self.angle_range)

        # Rotate along each axis separately
        image = scipy.ndimage.rotate(image, angle_x, axes=(1, 2), reshape=False, mode='nearest')
        image = scipy.ndimage.rotate(image, angle_y, axes=(0, 2), reshape=False, mode='nearest')
        image = scipy.ndimage.rotate(image, angle_z, axes=(0, 1), reshape=False, mode='nearest')

        return image

transforms = [
    Window(-200, 1000),
    MinMaxNorm(-200, 1000),
    RandomTranslation(max_shift=10),
    RandomFlip(),
    RandomRotation(angle_range=10)
]





class RibFractureDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_name = os.path.join(self.root_dir, str(self.data_frame.loc[idx, "Image_Name"]))
        image = nib.load(img_name).get_fdata()
        
        if self.transform:
            for t in self.transform:
                image = t(image)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        # Extract metadata
        has_fracture = self.data_frame.loc[idx, "Type (Displacement)"] != "No Fracture"
        location = self.data_frame.loc[idx, "Location"].lower().replace(" fracture", "")
        fracture_type = self.data_frame.loc[idx, "Type (Displacement)"]
        characterisation = self.data_frame.loc[idx, "Fracture Characterisation"]
        rib_side = self.data_frame.loc[idx, "Rib Side"].lower() if has_fracture else None

        # Generate improved description
        if has_fracture:
            fracture_type_map = {
                "Displaced/Offset": "displaced",
                "Non Displaced": "non-displaced",
                "Severely Displaced": "severely displaced",
                "Buckle": "buckle"
            }
            fracture_type_clean = fracture_type_map.get(fracture_type, fracture_type.lower())
            characterisation_clean = characterisation.lower().replace("linear ", "")
            description = (
                f"A {fracture_type_clean} rib fracture, characterized as {characterisation_clean}, "
                f"is located at the {location} aspect of the {rib_side} side."
            )
        else:
            description = "No rib fracture is detected in this CT patch."

        # Labels
        labels = {
            "head1": torch.tensor([self.data_frame.loc[idx, "Location_"]], dtype=torch.float32),
            "head2": torch.tensor([self.data_frame.loc[idx, "Type_"]], dtype=torch.float32),
            "head5": torch.tensor([self.data_frame.loc[idx, "Multiple_Fractures_"]], dtype=torch.float32),
            "head6": torch.tensor([self.data_frame.loc[idx, "Fracture_Characterization_"]], dtype=torch.float32),
        }

        return image, description, labels

# Example usage
train_csv_file = "/workspace/ribfrac/MICCAI-2025/RibFrac_Train/RibFrac_Class_Train.csv"
train_root_dir = '/workspace/ribfrac/MICCAI-2025/RibFrac_Train/images'
test_csv_file = "/workspace/ribfrac/MICCAI-2025/Detection_Model/Faster_Rcnn/patches/test_ribfrac.csv"
test_root_dir = '/workspace/ribfrac/MICCAI-2025/Detection_Model/Faster_Rcnn/patches/ct_patches'


# Assuming 'transforms' is defined elsewhere
train_dataset = RibFractureDataset(csv_file=train_csv_file, root_dir=train_root_dir, transform=transforms)
test_dataset = RibFractureDataset(csv_file=test_csv_file, root_dir=test_root_dir, transform=transforms)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Base UNet with ResNet50 encoder
        unet = smp.Unet(
            encoder_name="resnet50",
            in_channels=1,
            classes=1
        )
        
        self.encoder = unet.encoder
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        
        # Fully connected layers for each head
       
        self.head1 = nn.ModuleList([nn.Linear(self.encoder.out_channels[-1], 1) for _ in range(4)])
      
        self.head2 = nn.ModuleList([nn.Linear(self.encoder.out_channels[-1], 1) for _ in range(5)])
   
       
        self.head3 = nn.ModuleList([nn.Linear(self.encoder.out_channels[-1], 1) for _ in range(4)])
     
        self.head4 = nn.ModuleList([nn.Linear(self.encoder.out_channels[-1], 1) for _ in range(5)])
        
        

    def get_embeddings(self,x):
        # Extract features
        features = self.encoder(x)
        x = features[-1]
        x = self.pool(x)
        return self.flatten(x)
        
    def get_predictions(self,x):
        embeddings = self.get_embeddings(x)
        # Process through each head and subhead
        output = {}
        
        output['head1'] = [head(embeddings) for head in self.head1]
        output['head2'] = [head(embeddings) for head in self.head2]
        output['head3'] = [head(embeddings) for head in self.head5]
        output['head4'] = [head(embeddings) for head in self.head6]

        return output

    def forward(self, x):
        embeddings = self.get_embeddings(x)
        predictions = self.get_predictions(x)
        return predictions




from transformers import AutoTokenizer, AutoModel
import torch

class ClinicalBERTEmbedder:
    def __init__(self, model_name="abhinand/MedEmbed-large-v0.1"):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_cls_embedding(self, text, max_length=512,device="cuda"):
        
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        
        # Pass the tokenized text through the model
        outputs = self.model(**inputs)
        
        # Extract the CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]
        return cls_embedding


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256 ,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x.to("cuda"))
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HyperBolic(nn.Module):
    def __init__(
        self,
        image_embedding=2048,
        text_embedding=1024,
        output_dim=512,
        curv_init=1.0,
        learn_curv=True,
        entail_weight=0.2,
    ):
        super().__init__()
        self.image_encoder = Model()
        self.text_encoder = ClinicalBERTEmbedder()
        
        # Projection heads
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        
        # Initialize curvature parameter (hyperboloid curvature will be -curv)
        self.curv = nn.Parameter(torch.tensor(curv_init).log(), requires_grad=learn_curv)
        
        # Restrict curvature parameter to prevent training instability
        self.curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }
        
        # Learnable scalars for unit norm before exponential map
        self.visual_alpha = nn.Parameter(torch.tensor(output_dim**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(output_dim**-0.5).log())
        
        self.entail_weight = entail_weight
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))  # temperature parameter
        
    def exp_map0(self, x, curv, eps=1e-8):
        """Exponential map from tangent space at origin to hyperboloid."""
        rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
        sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
        return torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
    
    def pairwise_dist(self, x, y, curv, eps=1e-8):
        """Compute pairwise geodesic distance on hyperboloid."""
        x_time = torch.sqrt(1/curv + torch.sum(x**2, dim=-1, keepdim=True))
        y_time = torch.sqrt(1/curv + torch.sum(y**2, dim=-1, keepdim=True))
        
        # Compute Lorentzian inner product using matmul for proper broadcasting
        xy_space = torch.matmul(x, y.transpose(-2, -1))
        time_components = torch.matmul(x_time, y_time.transpose(-2, -1))
        inner_product = xy_space - time_components.squeeze(-1)
        
        # Compute distance
        c_xyl = -curv * inner_product
        return torch.acosh(torch.clamp(c_xyl, min=1 + eps)) / curv**0.5
    
    def oxy_angle(self, x, y, curv, eps=1e-8):
        """Compute exterior angle at x in hyperbolic triangle Oxy."""
        x_time = torch.sqrt(1/curv + torch.sum(x**2, dim=-1))
        y_time = torch.sqrt(1/curv + torch.sum(y**2, dim=-1))
        
        c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)
        
        acos_numer = y_time + c_xyl * x_time
        acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))
        
        acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
        return torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))
    
    def half_aperture(self, x, curv, min_radius=0.1, eps=1e-8):
        """Compute half aperture angle of entailment cone."""
        asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv**0.5 + eps)
        return torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))
    
    def encode_image(self, images, project=True):
        image_features = self.image_encoder.get_embeddings(images)
        image_features = self.image_projection(image_features)
        
        if project:
            image_features = image_features * torch.exp(self.visual_alpha)
            image_features = self.exp_map0(image_features, torch.exp(self.curv))
        
        return image_features
    
    def encode_text(self, texts, project=True):
        text_embeddings = []
        for text in texts:
            text_feature = self.text_encoder.get_cls_embedding(text).squeeze(0)
            text_projection = self.text_projection(text_feature)
            text_embeddings.append(text_projection)
        text_features = torch.stack(text_embeddings, dim=0)
        
        if project:
            text_features = text_features * torch.exp(self.textual_alpha)
            text_features = self.exp_map0(text_features, torch.exp(self.curv))
        
        return text_features

    def forward(self, batch):
        images = batch["images"].to("cuda")
        texts = batch["texts"]
        
        # Clamp parameters
        self.curv.data = torch.clamp(self.curv.data, **self.curv_minmax)
        _curv = torch.exp(self.curv)
        
        # Clamp scaling factors
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)
        
        # Get features
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        
        # Compute distances for contrastive loss
        distances = self.pairwise_dist(text_features, image_features, _curv)
        
        # Compute contrastive loss
        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        _scale = torch.exp(self.logit_scale)
        
        # Compute scaled logits
        logits = -distances * _scale
        
        # Create labels and ensure they're the right type
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=images.device)
        
        # Compute cross entropy loss with proper types
        contrastive_loss = (
            F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.transpose(-2, -1), labels)
        ) / 2
        
        # Compute hyperbolic entailment loss
        _angle = self.oxy_angle(text_features, image_features, _curv)
        _aperture = self.half_aperture(text_features, _curv)
        entailment_loss = torch.clamp(_angle - _aperture, min=0).mean()
        
        # Combine losses
        total_loss = contrastive_loss + self.entail_weight * entailment_loss
        
        return total_loss

train_csv_file = "/workspace/ribfrac/MICCAI-2025/RibFrac_Train/RibFrac_Class_Train.csv"
train_df = pd.read_csv(train_csv_file)

head_info = {
    'head1': {'column': 'Location_', 'n_classes': 4},
    'head2': {'column': 'Type_', 'n_classes': 5},
    'head3': {'column': 'Multiple_Fractures_', 'n_classes': 4},
    'head4': {'column': 'Fracture_Characterization_', 'n_classes': 5},
}

class_weights = {}
for head, info in head_info.items():
    column = info['column']
    n_classes = info['n_classes']
    # Get class distribution
    value_counts = train_df[column].value_counts().sort_index()
    # Ensure all classes are represented (handle missing classes)
    counts = value_counts.reindex(range(n_classes), fill_value=1).values  # Add minimal count to avoid division by zero
    # Calculate normalized reciprocal weights
    weights = 1.0 / counts
    weights = weights / weights.sum()  # Normalize
    class_weights[head] = torch.tensor(weights, dtype=torch.float32).to(device)

# Create loss functions with class weights
criterion = {
    head: nn.CrossEntropyLoss(weight=class_weights[head])
    for head in ['head1', 'head2', 'head3', 'head4']
}








# Data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# Initialize the model
model = Model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Load pretrained weights
checkpoint_path = "/workspace/ribfrac/MICCAI-2025/Classification_Model/Airrib_Model/rib_frac_model.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint, strict=True)


# CLIP model setup
clip_model = HyperBolic()
clip_model.to(device)
for param in clip_model.text_encoder.model.parameters():
    param.requires_grad = False

# Optimizer and loss setup
optimizer = optim.Adam([
    {'params': model.parameters()},
    {'params': clip_model.image_projection.parameters()},
    {'params': clip_model.text_projection.parameters()}
     
], lr=1e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)



embedder = ClinicalBERTEmbedder()
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    clip_model.train()
    running_loss = 0.0
    
    for images, texts, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
        images = images.to(device)
        labels = {key: value.to(device).squeeze().long() for key, value in labels.items()}
        text_batch = texts
        
        optimizer.zero_grad()
        clip_loss = clip_model({"images": images, "texts": text_batch})
        outputs = model(images)
        
        classification_loss = 0.0
        for head in ['head1', 'head2', 'head3', 'head4']:
            head_outputs = torch.stack(outputs[head], dim=1)
            head_outputs = head_outputs.squeeze(-1)
            head_labels = labels[head].view(-1)
          
            loss = criterion[head](head_outputs, head_labels) 
            
            classification_loss += loss
            
        
        combined_loss = classification_loss + 0.3* clip_loss
        combined_loss.backward()
        optimizer.step()
        running_loss += combined_loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}")
    scheduler.step()
    
   

# Save the model
torch.save(model.state_dict(), "rib_fracture_model_Meru_50.pth")
print("\nModel training complete and saved!")
