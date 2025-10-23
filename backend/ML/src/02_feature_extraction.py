# feature_extraction.py
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os
from data import load_preprocessed
import joblib

class NumpyImageDataset(Dataset):
    def __init__(self, X_np, transform=None):
        self.X = X_np
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img_np = self.X[idx]  # e.g. HWC uint8
        img = Image.fromarray(img_np)
        if self.transform:
            img = self.transform(img)
        return img

def extract_embeddings(X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model.eval()
    # remove final fc
    modules = list(model.children())[:-1]
    feat_extractor = torch.nn.Sequential(*modules).to(device)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    ds = NumpyImageDataset(X, transform=transform)
    dl = DataLoader(ds, batch_size=64, num_workers=4)
    emb_list = []
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device)
            feats = feat_extractor(batch)  # shape B x 512 x 1 x 1
            feats = feats.reshape(feats.size(0), -1).cpu().numpy()
            emb_list.append(feats)
    embeddings = np.vstack(emb_list)
    joblib.dump(embeddings, "models/embeddings.joblib")
    return embeddings
