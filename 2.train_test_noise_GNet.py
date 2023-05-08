# -*- coding: utf-8 -*-
"""test_clean_GNet.ipynb
"""

import pandas as pd

df = pd.read_csv('/home/subrat/JoCoR-env/archive/HAM10000_metadata.csv')

print(df)

#Using os library to walk through folders
import os
import cv2
from skimage import img_as_ubyte

img_number = 1 # not mandatory to write this line
for root, dirs, files in os.walk("archive/HAM10000_images_part_1"):
    
#for path,subdir,files in os.walk("."):
#   for name in dirs:
#       print (os.path.join(root, name)) # will print path of directories
   for name in files:    
       print (os.path.join(root, name)) # will print path of files 
       path = os.path.join(root, name)
       img= cv2.imread(path, 0)  #now, we can read each file since we have the full path

#dropping bkl and bcc classes
in_dist_True = df[df.dx != 'bkl']
in_dist_True = in_dist_True[in_dist_True.dx != 'bcc']
# p=df.drop(df.iloc[df.dx == 'bkl'])

in_dist_True.dx.hist()

len(in_dist_True)

# save one_hot _encoded file as a csv file in 'archive' folder
in_dist_True.to_csv('archive/test_true_data_encd.csv', index=False)

pd.read_csv('archive/test_true_data_encd.csv')

dataf = pd.read_csv('/home/subrat/JoCoR-env/archive/test_true_data_encd.csv')
print(dataf)

# Converting differnt label to one hot encoding
print(len(dataf))
for i in range(len(dataf)):
    if dataf['dx'][i]== 'nv':
        dataf.at[i, 'dx']= 0
        
    if dataf['dx'][i]== 'df':
        dataf.loc[i, 'dx'] = 1
        
    if dataf['dx'][i]== 'vasc':
        dataf.loc[i, 'dx'] = 2
        
    if dataf['dx'][i] == 'mel':
        dataf.loc[i, 'dx'] = 3
    
    if dataf['dx'][i]== 'akiec':
        dataf.loc[i, 'dx'] = 4

# save one_hot _encoded file as a csv file in 'archive' folder
dataf.to_csv('archive/True_test_data_encd.csv', index=False)

d = pd.read_csv('./archive/True_test_data_encd.csv')

print(d)



#d.to_csv('archive/TestF.csv')

#pd.read_csv('archive/TestF.csv')

for i in range(len(d)):
    b = str(dataf['image_id'][i])
    a = os.path.join('/home/subrat/JoCoR-env/archive/HAM10000_images_part_1/',b)
    print(a)
    d.at[i, 'image_id'] = a

# dataf.assign(
#     dx_type=
#     dataf[['dx']].to_numpy().argmax(axis=1) +1
# )
# d['dy'] = d['dx']



d=d.drop(['lesion_id', 'dx_type', 'localization', 'sex','age'], axis=1)
d.to_csv('Test_TRUE_encoded', index=False)
pd.read_csv('Test_TRUE_encoded')

label_T = d['dx']

label_T.shape

d['label_T'] = label_T

d['label_F'] = label_T

d=d.drop('dx', axis=1)

d

d['label_F'].hist()

import random

#  symetric noise of 20% between the bellow classes
mel_id = d.index[d.label_F == 3].tolist()
nv_id = d.index[d.label_F == 0].tolist() # mel =3

#  symetric noise of 10% between the bellow classes
df_id = d.index[d.label_F == 1].tolist()
vasc_id = d.index[d.label_F == 2].tolist()

in_dist_Noisy_clean = d
print("Initialisation done for in_dist_Noisy")

k_mel = int(len(mel_id)*0.1) # select 20% from the mel class

# print(len(mel_id))
print(k_mel)
mel_noisy_id = random.sample(mel_id,k_mel) # sample randomly k_mel length of elements from the mel_id sequence. 20% of the list
print(len(mel_noisy_id) == k_mel)

for idx in mel_noisy_id:
    in_dist_Noisy_clean.at[idx, 'label_F'] = 0
    
print("20% of the 3 label has been converted to 0")
# print(f'the new mel count is {len(in_dist_Noisy.loc[in_dist_Noisy.dx == 'mel'])}')
    
k_nv = int(len(nv_id)*0.1)
print(k_nv)
nv_noisy_id = random.sample(nv_id,k_nv)
print(len(nv_noisy_id)==k_nv)

for idx in nv_noisy_id:
    in_dist_Noisy_clean.at[idx, 'label_F'] = 3

print("20% of the 0 label has been converted to 3")
# print(f'the new mel count is {len(in_dist_Noisy.loc[in_dist_Noisy.dx == 'nv'])}')
    
    
k_df = int(len(df_id)*0.2)
print(k_df)
df_noisy_id = random.sample(df_id,k_df)
print(len(df_noisy_id)==k_df)

print()
for idx in df_noisy_id:
    in_dist_Noisy_clean.at[idx, 'label_F'] = 2

print("10% of the 1 label has been converted to 2")
# print(f'the new mel count is {len(in_dist_Noisy.loc[in_dist_Noisy.dx == 'df'])}')
    
k_vasc = int(len(vasc_id)*0.2)
print(k_vasc)
vasc_noisy_id = random.sample(vasc_id,k_vasc)
print(len(vasc_noisy_id)==k_vasc)

for idx in vasc_noisy_id:
    in_dist_Noisy_clean.at[idx, 'label_F'] = 1

    
print("10% of the 2 label has been converted to 1")
# print(f'the new mel count is {len(in_dist_Noisy.loc[in_dist_Noisy.dx == 'vasc'])}')

in_dist_Noisy_clean.label_F.hist()

# D=in_dist_Noisy.to_csv('archive/labelF_train_data.csv', index=False)

in_dist_Noisy_clean.to_csv('archive/Noisy_train_data_clean.csv', index=False)

import pandas as pd
dff = pd.read_csv('./archive/Noisy_train_data_clean.csv')

dff['image_id'][0]

# for i in range(len(dff)):
#     b = str(df['image_id'][i])
#     a = os.path.join('/home/subrat/JoCoR-env/archive/HAM10000_images_part_1/',b)
#     print(a)
#     dff.at[i, 'image_id'] = a

# from skimage import io
# image = io.imread('/home/subrat/JoCoR-env/archive/HAM10000_images_part_1/ISIC_0030015.jpg')
# from matplotlib import pyplot as plt
# plt.imshow(image)

from skimage import io
image = io.imread('/home/subrat/JoCoR-env/archive/HAM10000_images_part_1/ISIC_0030011.jpg')
from matplotlib import pyplot as plt
plt.imshow(image)


root_dir = '/home/subrat/JoCoR-env/archive/HAM10000_images_part_1/'
csv_file = '/home/subrat/JoCoR-env/archive/Noisy_train_data_clean.csv'



import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
import os
import pandas as pd
from skimage import io
from torch.utils.data import Dataset,DataLoader  # Gives easier dataset managment and creates mini batches


class HAM10000(Dataset):
    def __init__(self, csv_file="/home/subrat/JoCoR-env/archive/Noisy_train_data_clean.csv", transform=None):
        self.csv_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index):
        label_true=self.csv_data.loc[index, 'label_T']
        label_false=self.csv_data.loc[index, 'label_F']
        img_path = self.csv_data.loc[index, 'image_id']+'.jpg'
#         img_path = os.path.join(root_dir, (annotation.iloc[index, 1] + '.jpg'))
        image = io.imread(img_path)
#         y_label = torch.tensor(int(self.annotations.iloc[index, 2]))
        
        if self.transform:
            image = self.transform(image)

        return (image, label_true, label_false)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channel = 3
num_classes = 5
learning_rate = 1e-3
batch_size = 16
num_epochs = 1

# Load Data
transform=transforms.ToTensor()
dataset = HAM10000(transform=transform)

# Dataset is actually a lot larger ~25k images, just took out 10 pictures
# to upload to Github. It's enough to understand the structure and scale
# if you got more images.

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
# train_set, test_set = torch.utils.data.random_split(dataset, train_set[0],test_set[0])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
# print(train_loader.label)
print(f'No of batch loaded for training: {len(train_loader)}')
# Model
model = torchvision.models.googlenet(pretrained=True)
model.to(device)
#print(model)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, batch in enumerate(train_loader):
        # Get data to cuda if possible
#         print(batch_idx, batch[0].size(),
#           batch[1].size())
        print(f'data is :', data)

        data = batch[0]
        targets = batch[1]
        data = data.to(device=device)
        targets = targets.to(device=device)
#         
        # forward
        output = model(data)
        loss = criterion(output, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")

# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_incorrect = 0
    num_samples = 0
    model.eval()
    

    with torch.no_grad():
        for x, yT, yF in loader:
            x = x.to(device=device)
            yT = yT.to(device=device)
            yF = yF.to(device=device)
            #print(yT.numpy()) 
           

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == yT).sum()
            num_incorrect += (predictions == yF).sum()
            num_samples += predictions.size(0)

        print(
             f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

        model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)







