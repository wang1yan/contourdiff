import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
from glob import glob
from sklearn.model_selection import train_test_split

class MulticueEdgeDataset(Dataset):
    def __init__(self, args, dataset_dir, transform = None, mode = 'Training',plane = False):
        # Set dataset directory and split.
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.transform = transform
        
        # Read the list of images and (possible) edges.
        if self.mode == 'Training':
            self.list_path = os.path.join(self.dataset_dir , 'image-train.lst')
        else:  # Assume test.
            self.list_path = os.path.join(self.dataset_dir , 'image-test.lst')
        with open(self.list_path, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]  # Remove the newline at last.
        self.images_name = []
        if self.mode == 'Training':
            pairs = [line.split() for line in lines]
            self.images_path = [pair[0] for pair in pairs]
            self.edges_path = [pair[1] for pair in pairs]
            for image_path in self.images_path:
                element=image_path[20:]
                self.images_name.append(element[:len(element)-4])
        else:
            self.images_path = lines
            self.images_name = []  # Used to save temporary edges.
            for path in self.images_path:
                folder, filename = os.path.split(path)
                name, ext = os.path.splitext(filename)
                self.images_name.append(name)


    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        edge = None
        if self.mode == "Training":
            # Get edge.
            edge_path = os.path.join(self.dataset_dir, self.edges_path[index])
            #name = self.images_name[index]
            edge = Image.open(edge_path).convert('L')
            if self.transform:
                edge = self.transform(edge)
        # Get image.
        image_path = os.path.join(self.dataset_dir, self.images_path[index])
        name = self.images_name[index]
        image = Image.open(image_path).convert('RGB')
        # 如果提供了转换函数，就对图像和掩码进行预处理
        if self.transform:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            
    
        # Return image and edge.
        if self.mode == 'Training':
            return (image, edge, name)
        else:
            return (image,name)