import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import torchvision
from torchvision.models.resnet import ResNet101_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import math
import matplotlib.pyplot as plt
import random
from torchvision.models.resnet import ResNeXt101_32X8D_Weights

class Window:
    def __init__(self, window_min, window_max):
        self.window_min = window_min
        self.window_max = window_max

    def __call__(self, image):
        # Clip image pixel values to within the specified window
        image = np.clip(image, self.window_min, self.window_max)
        return image

class MinMaxNorm:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, image):
        # Normalize image pixel values to range [0, 1]
        image = (image - self.low) / (self.high - self.low)
        return image

class FractureDetect(Dataset):
    def __init__(self, img_dir, dataframe, num_classes=1, transforms=None, augmentation=None):
        self.image_ids = dataframe["image_name"].unique()
        self.img_dir = img_dir
        self.transforms = transforms
        self.augmentation = augmentation
        self.num_classes = num_classes + 1
        self.df = dataframe

    def __len__(self):
        return len(self.image_ids)

    def _apply_transforms(self, image):
        for t in self.transforms:
            image = t(image)
        return image

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_name'] == image_id].reset_index(drop=True)
        
        img = nib.load(os.path.join(self.img_dir, image_id))
        img = img.get_fdata().astype(float)
        
        
        
        # Apply windowing and normalization
        if self.transforms is not None:
            image_arr = self._apply_transforms(img)
        
        # Prepare the image array for 3-channel input (for ResNet backbone)
        image_arr = np.stack([image_arr, image_arr, image_arr], axis=-1)

        # Load bounding boxes and labels
        boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values
        labels = records["label"].values

        # Prepare targets
        if boxes is not None and len(boxes) > 0:
            labels = torch.tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            area = torch.as_tensor(area, dtype=torch.float32)
            bboxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            labels = torch.tensor([], dtype=torch.int64)
            bboxes = torch.tensor([]).reshape(0, 4)  # Ensure shape [0, 4]
            area = torch.tensor([], dtype=torch.float32)

        iscrowd = torch.zeros((labels.size(0),), dtype=torch.int64)

        # Apply augmentations if any
        if self.augmentation:
            try:
                augmented = self.augmentation(image=image_arr, bboxes=bboxes.tolist(), class_labels=labels.tolist())
                image_arr = augmented['image']
                bboxes = torch.tensor(augmented['bboxes'], dtype=torch.float32)
                labels = torch.tensor(augmented['class_labels'], dtype=torch.int64)
            except ValueError as e:
                print(f"Skipping image {image_id} due to error: {e}")
                return self.__getitem__((index + 1) % len(self))
            
            if bboxes.size(0) == 0:
                return self.__getitem__((index + 1) % len(self))
            
        target = {
            'boxes': bboxes,
            'labels': labels,
            'area': area,
            'image_id': torch.tensor([index]),
            'iscrowd': iscrowd
        }

        image_arr = torch.tensor(image_arr, dtype=torch.float).permute(2, 0, 1)

        return image_arr, target

    @staticmethod
    def collate_fn(data):
        return tuple(zip(*data))

    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle=False, num_workers=0):
        return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=FractureDetect.collate_fn)


train_df = pd.read_csv("/workspace/ribfrac/MICCAI-2025/AiRribRibFrac_slice/Train/Train.csv")
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
train_img_dir = '/workspace/ribfrac/MICCAI-2025/AiRribRibFrac_slice/Train/images'

# Load and shuffle the validation DataFrame
val_df = pd.read_csv("/workspace/ribfrac/MICCAI-2025/AiRribRibFrac_slice/Test/bounding_boxes.csv")
val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_img_dir = '/workspace/ribfrac/MICCAI-2025/AiRribRibFrac_slice/Test/images'

transforms = [
    Window(-200, 1000),
    MinMaxNorm(-200, 1000),
]

# train_data = FractureDetect(train_img_dir, train_df, transforms=transforms, augmentation=augmentation_pipeline)
train_data = FractureDetect(train_img_dir, train_df, transforms=transforms)
val_data = FractureDetect(val_img_dir, val_df, transforms=transforms)

train_loader = FractureDetect.get_dataloader(train_data, 8, False)
val_loader = FractureDetect.get_dataloader(val_data, 8, False)

# Define model
anchor_sizes = ((4,), (8,), (16,), (32,), (64,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

rpn_anchor_generator = AnchorGenerator(
    anchor_sizes, aspect_ratios
)


backbone = resnet_fpn_backbone(backbone_name='resnext101_32x8d', weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
model = FasterRCNN(backbone=backbone, num_classes=2, rpn_anchor_generator=rpn_anchor_generator)


pretrained_path = '/workspace/ribfrac/MICCAI-2025/Detection_Model/Faster_Rcnn/best_model_rexnext101_1.pth'  # Update this path
if os.path.exists(pretrained_path):
    print(f"Loading pretrained weights from {pretrained_path}")
    state_dict = torch.load(pretrained_path)
    model.load_state_dict(state_dict)
    print("Pretrained weights loaded successfully")
else:
    print("No pretrained weights found, starting from ImageNet weights")




params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=1e-4, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

train_len = len(train_loader)
val_len = len(val_loader)

model = model.cuda()
min_val_loss = math.inf

for epoch in range(35):
    train_loss_epoch = 0
    val_loss_epoch = 0
    start = time.time()
    print("train_start")
    model.train()
    for images, targets in train_loader:
        images = list(image.cuda().float() for image in images)
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        train_loss_dict = model(images, targets)
        train_losses = sum(loss for loss in train_loss_dict.values())


        optimizer.zero_grad()
        train_losses.backward()
        optimizer.step()
        train_loss_epoch += train_losses.item()

    train_loss_epoch = train_loss_epoch / train_len

   
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(image.cuda().float() for image in images)
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            val_loss_dict = model(images, targets)
            val_losses = sum(loss for loss in val_loss_dict.values())
            val_loss_epoch += val_losses.item()

    val_loss_epoch = val_loss_epoch / val_len
    end = time.time()
    time_ = (end - start) / 60
    print(f'Epoch: {epoch}/50   Train_loss: {train_loss_epoch:.6f}    Val_loss: {val_loss_epoch:.6f}  Time: {time_:.6f}')
    checkpoint_path = f'model_checkpoints_airrib/last_model_resnext101__{epoch}.pth'
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Model checkpoint saved: {checkpoint_path}')
    lr_scheduler.step(val_loss_epoch)
    if min_val_loss > val_loss_epoch:
        min_val_loss = val_loss_epoch
        torch.save(model.state_dict(), 'best_model_rexnext101_airrib.pth')
        print(f'Best model saved at epoch {epoch}')

torch.save(model.state_dict(), 'last_model_resnext101_airrib.pth')
