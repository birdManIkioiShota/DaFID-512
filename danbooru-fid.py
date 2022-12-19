import argparse
import glob
import numpy as np
import os
from PIL import Image
from scipy import linalg
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.transformers = transforms.Compose([
            transforms.Resize(360),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979]),
        ])

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, i):
        img = Image.open(self.file_list[i]).convert('RGB')
        img = self.transformers(img)
        return img

def compute_mu_and_sigma(path, model, batch_size, num_workers,dims, device):
    file_list = glob.glob(os.path.join(path, '*'))
    
    model.eval()

    dataset = ImageDataset(file_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    embeddings = np.empty((len(file_list), dims))
    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).cpu().numpy()
        
        embeddings[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    mu = np.mean(embeddings, axis=0)
    sigma = np.cov(embeddings, rowvar=False)

    return mu, sigma

def main(opt):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('RF5/danbooru-pretrained', 'resnet50').to(device)

    if opt.dims == 512:
        identity_falyers = [7, 8]
    elif opt.dims == 4096:
        identity_falyers = [3, 4, 5, 6, 7, 8]
    else:
        identity_falyers = []

    for i in identity_falyers:
        model[1][i] = nn.Identity()
    
    # print(model)

    mu_real, sigma_real = compute_mu_and_sigma(opt.A, model, opt.batch_size, opt.num_workers, opt.dims, device)
    mu_fake, sigma_fake = compute_mu_and_sigma(opt.B, model, opt.batch_size, opt.num_workers, opt.dims, device)

    covmean, _ = linalg.sqrtm(sigma_fake.dot(sigma_real), disp=False)
    if np.iscomplexobj(covmean): covmean = covmean.real

    diff = mu_real - mu_fake

    fid = (diff.dot(diff) + np.trace(sigma_fake) + np.trace(sigma_real) - 2 * np.trace(covmean))
    print(fid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--A', type=str, required=True)
    parser.add_argument('--B', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--dims', type=int, choices=[4096, 512, 6000], default=512)
    opt = parser.parse_args()

    main(opt)