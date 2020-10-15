from torch.utils.data import Dataset, random_split
import math
from PIL import Image
import cv2
import numpy as np
import torch
import os
from tqdm import notebook
# -----------------------------------------------------Main Function which calls everything--------------------------------------------------------------
def RawDataSet(train_split = 70,test_transforms = None,train_transforms = None, set_no=1, url_path ='None', whole_data = True ):

  dataset = Rawdata(url=url_path,set_no = set_no, whole_data = whole_data )
  train_len = len(dataset)*train_split//100
  test_len = len(dataset) - train_len 
  print(len(dataset))
  train_set, val_set = random_split(dataset, [train_len, test_len])
  train_dataset = DatasetFromSubset(train_set, transform=train_transforms)
  test_dataset = DatasetFromSubset(val_set, transform=train_transforms)
  del dataset
  del train_len
  del test_len
  del train_set
  del val_set
  
  return train_dataset, test_dataset



# --------------------------------------------------------------Custom data set-------------------------------------------------------------------------

class Rawdata(Dataset):
    def __init__(self,url, set_no, whole_data):
        self.fg_bg = []
        self.depth = []
        self.bg = []
        self.url = url
        self.mask = []
        
        if(whole_data):

          for set_no in range(1,6):
            pref_url = '/content/data_'+str(set_no)
            for i in notebook.tqdm(range(80000)):
            
              self.fg_bg.append(f'{pref_url}/Fg-Bg/fg-bg{str(i+1 + (set_no-1)*80000)}.jpg')  
              self.bg.append(math.ceil((i+1)/800)) #80k images spanned over 100 bg's. Means 800 images per background
              self.depth.append(f'{pref_url}/Depth/depth{str(i+1 + (set_no-1)*80000)}.jpg')
              self.mask.append(f'{pref_url}/Fg-Bg-Mask/fg-bg-mask{str(i+1 + (set_no-1)*80000)}.jpg')

      
        else:
            for i in notebook.tqdm(range(80000)):
              
                self.fg_bg.append(f'{url}/Fg-Bg/fg-bg{str(i+1 + (set_no-1)*80000)}.jpg')  
                self.bg.append(math.ceil((i+1)/800)) #80k images spanned over 100 bg's. Means 800 images per background
                self.depth.append(f'{url}/Depth/depth{str(i+1 + (set_no-1)*80000)}.jpg')
                self.mask.append(f'{url}/Fg-Bg-Mask/fg-bg-mask{str(i+1 + (set_no-1)*80000)}.jpg')
        # print(f'{url}/Fg-Bg/fg-bg{str(i+1 + (set_no-1)*80000)}.jpg')
            
    def __len__(self):
        return len(self.fg_bg)

    def __getitem__(self, idx):
        fg_bg = self.fg_bg[idx]
        depth = self.depth[idx]
        bg = self.bg[idx]
        mask = self.mask[idx]
        return fg_bg,bg,mask,depth



# ----------------------------------------------------Data subset which comes after splitting--------------------------------------------------

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        

    def __getitem__(self, index):
        fg_bg,bg,mask,depth = self.subset[index]
        
       
        fg_bg=cv2.imread(fg_bg)[:, :, [2, 1, 0]]
        
        bg = cv2.imread(f"/content/gdrive/My Drive/Mask_Rcnn/Background/bg{str(bg)}.jpg")[:, :, [2, 1, 0]]
        #print(fg_bg,bg)
        inputimg = np.concatenate((bg,fg_bg ), axis=2)
        mask = cv2.imread(mask,cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(depth,cv2.IMREAD_GRAYSCALE)
        
        
        


        if self.transform:

            #print(transform)
            inputimg = self.transform['ip'](inputimg)
            mask = self.transform['op'](mask)
            depth = self.transform['op'](depth)
        return inputimg, mask,depth

    def __len__(self):
        return len(self.subset)

# -------------------------------------------------------------------------------------------------------------------------------------------------------




