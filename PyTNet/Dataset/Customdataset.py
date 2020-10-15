"""tinyimagenet.py

"""

from torch.utils.data import Dataset, random_split
from PIL import Image
import numpy as np
import torch
import os
import torchvision.transforms as transforms
from tqdm import notebook
import zipfile


# -----------------------------------------------------Main Function which calls everything--------------------------------------------------------------
def CustomDataSet(train_split = 70,test_transforms = None,train_transforms = None, path = "/content/dataset/"):




  dataset = TinyImageNet(path)
  train_len = len(dataset)*train_split//100
  test_len = len(dataset) - train_len 
  train_set, val_set = random_split(dataset, [train_len, test_len])
  train_dataset = DatasetFromSubset(train_set, transform=train_transforms)
  test_dataset = DatasetFromSubset(val_set, transform=test_transforms)

  return train_dataset, test_dataset



# --------------------------------------------------------------Custom data set-------------------------------------------------------------------------

class TinyImageNet(Dataset):
    def __init__(self,path ='/content/dataset/' ):
        self.data = []
        self.target = []
        self.classes = {'Flying_Birds' : 0,  'Large_QuadCopters' : 1 , 'Small_QuadCopters' : 2 , 'Winged_Drones' : 3}
        for classs in self.classes : 
          full_path = path + classs
          for filename in os.listdir(full_path):
            self.data.append(full_path+"/"+filename)
            self.target.append(self.classes[classs])
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.target[idx]
        img = data     
        
        return data,target



# ----------------------------------------------------Data subset which comes after splitting--------------------------------------------------

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
       
        x = Image.open(x).convert('RGB')
      
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# -------------------------------------------------------------------------------------------------------------------------------------------------------

