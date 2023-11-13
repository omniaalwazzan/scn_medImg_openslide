#%%
import os, os.path
from PIL import Image
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import random
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os 
from torchvision import transforms
import pandas as pd
#%%
df = pd.read_csv(r"D:/data/img/data.csv",index_col=(0))
df['GT'], label_mapping = pd.factorize(df['GT'])
df = df.rename(columns={'ID': 'Patient ID'})
df = df.rename(columns={'GT': 'label'})

class histoDataset(Dataset):

    def __init__(self, df, transform, label):
        
        self.transform = transform 
        self.labels = df[label].astype(int).tolist()
        self.filepaths = df['ImagePath'].tolist()
        #self.stain = df['Stain'].tolist()
        #self.patient_ID = df['Patient ID'].tolist()
        #self.filename = df['Filename'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        try:
            image = Image.open(self.filepaths[idx])
            #patient_id = self.patient_ID[idx]
            #filename = self.filename[idx]
            #stain = self.stain[idx]
            image_tensor = self.transform(image)
            image_label = self.labels[idx]
            return image_tensor, image_label#, patient_id, filename, stain
        except FileNotFoundError:
            return None

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# Create an instance of the custom dataset
label = "label"

dataset = histoDataset(df=df, transform=transform,label = label)
dataset.filepaths
# Create a data loader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
a = next(iter(data_loader))
# %%
def collate_fn_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) 
# %%

class Loaders:
    def train_test_ids(self, df, train_fraction, random_state, patient_id, label, subset=False):
        
        # patients need to be strictly separated between splits to avoid leakage. 
        ids  = df[patient_id].tolist()
        file_ids = sorted(set(ids))
    
        train_ids, test_ids = train_test_split(file_ids, test_size=1-train_fraction, random_state=random_state)
        
        if subset:
            
            train_subset_ids = random.sample(train_ids, 10)
            test_subset_ids = random.sample(test_ids,5)
            
            return file_ids, train_subset_ids, test_subset_ids
        
        return file_ids, train_ids, test_ids

    def df_loader(self, df, train_transform, test_transform, train_ids, test_ids, patient_id, label, subset=False):
        
        train_subset = df[df[patient_id].isin(train_ids)].reset_index(drop=True)
        test_subset = df[df[patient_id].isin(test_ids)].reset_index(drop=True)
       # df_train = histoDataset(train_subset, train_transform, label=label)
        #df_test = histoDataset(test_subset, test_transform, label=label)
        
        return train_subset, test_subset#, df_train, df_test, 


    def slides_dataloader(self, train_sub, test_sub, train_ids, test_ids, train_transform, test_transform, slide_batch, num_workers, shuffle, collate, label='Pathotype_binary', patient_id="Patient ID"):
        
        # TRAIN dict
        train_subsets = {}
        for i, file in enumerate(train_ids):
            new_key = f'{file}'
            train_subset = histoDataset(train_sub[train_sub["Patient ID"] == file], train_transform, label=label)
#            if len(train_subset) != 0:
            train_subsets[new_key] = torch.utils.data.DataLoader(train_subset, batch_size=slide_batch, shuffle=shuffle, num_workers=num_workers, drop_last=False, collate_fn=collate)

            
        # TEST dict
        test_subsets = {}
        for i, file in enumerate(test_ids):
            new_key = f'{file}'
            test_subset = histoDataset(test_sub[test_sub["Patient ID"] == file], test_transform, label=label)
#            if len(test_subset) != 0:
            test_subsets[new_key] = torch.utils.data.DataLoader(test_subset, batch_size=slide_batch, shuffle=shuffle, num_workers=num_workers, drop_last=False, collate_fn=collate)
        
        return train_subsets, test_subsets
    

# %%
train_fraction = 0.7
seed = 42
patient_id = "Patient ID"
label = "label"
subset = False
slide_batch = 10
num_workers = 0

# Image transforms # TODO
train_transform = transforms.Compose([
        transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.1),
        transforms.ColorJitter(contrast=0.1),
        transforms.ColorJitter(saturation=0.1),
        transforms.ColorJitter(hue=0.1)]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
# %%

file_ids, train_ids, test_ids = Loaders().train_test_ids(df, train_fraction, seed, patient_id, label, subset)
#%%
train_subset, test_subset = Loaders().df_loader(df, train_transform, test_transform, train_ids, test_ids, patient_id, label, subset=subset)
#%%
train_slides, test_slides = Loaders().slides_dataloader(train_subset, test_subset, train_ids, test_ids, train_transform, test_transform, slide_batch=slide_batch, num_workers=num_workers, shuffle=False, collate=collate_fn_none, label=label, patient_id=patient_id)
a = next(iter(train_slides['NH05-418']))

# %%

import torch.nn as nn
from torchvision import models
from torchvision.models import vgg16_bn, VGG16_BN_Weights


class VGG_embedding(nn.Module):

    """
    VGG16 embedding network for WSI patches
    """

    def __init__(self, embedding_vector_size=1024, n_classes=24):

        super(VGG_embedding, self).__init__()

        embedding_net = models.vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)

        # Freeze training for all layers
        for param in embedding_net.parameters():
            param.require_grad = False

        # Newly created modules have require_grad=True by default
        num_features = embedding_net.classifier[6].in_features
        features = list(embedding_net.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, embedding_vector_size)])
        features.extend([nn.Dropout(0.5)])
        features.extend([nn.Linear(embedding_vector_size, n_classes)]) # Add our layer with n outputs
        embedding_net.classifier = nn.Sequential(*features) # Replace the model classifier

        features = list(embedding_net.classifier.children())[:-2] # Remove last layer
        embedding_net.classifier = nn.Sequential(*features)
        self.vgg_embedding = nn.Sequential(embedding_net)

    def forward(self, x):

        output = self.vgg_embedding(x)
        output = output.view(output.size()[0], -1)
        return output
model = VGG_embedding()
# %%
import torchvision
from torchvision import models
from torch import nn
import torch
from torchsummary import summary
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class convNext(nn.Module):
    def __init__(self, n_classes=24):
        super().__init__()
        #convNext = models.convnext_base(pretrained=True)
        convNext = models.convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        for param in convNext.parameters():
            param.require_grad = False
        feature_extractor = nn.Sequential(*list(convNext.children())[:-1])
        self.feature = feature_extractor
        self.calssifier =nn.Sequential(nn.Flatten(1, -1),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(in_features=1024, out_features=n_classes))

    def forward(self, x):
        feature = self.feature(x) # this feature we can use when doing stnad.Att
        flatten_featur = feature.view(feature.size(0), -1) #this we need to plot tsne
        x = self.calssifier(feature)
        return flatten_featur
model = convNext()
# %%
for batch_idx, loader in enumerate(train_slides.values()):
    print("\rTraining batch {}/{}\n".format(batch_idx, len(train_slides)), end='', flush=True)
    patient_embedding = []
    for data in loader:
        inputs, label = data
        print(inputs.shape)
        
        
# %%
for batch_idx, (patient_ID, embedding) in enumerate(train_slides.items()):
    for data in embedding:
        inputs, label = data
        print(inputs.shape)
# %%


import os, os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import time

from collections import Counter
from collections import defaultdict

from PIL import Image
from PIL import ImageFile

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms, models


from training_loops import train_embedding, train_att_slides, test_slides, soft_vote

from attention_models import VGG_embedding, GatedAttention

from plotting_results import auc_plot, pr_plot, plot_confusion_matrix

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

plt.ion()  

import gc 
gc.enable()
#%%

# %%

torch.manual_seed(42)
train_fraction = .7
random_state = 2

subset= False

train_batch = 10
test_batch = 1
slide_batch = 1

num_workers = 0
shuffle = False
drop_last = False

train_patches = True
train_slides = True
testing_slides = True

embedding_vector_size = 1024

subtyping = True # (True for 3 class problem) 

# %%

stain = 'CD138'

# %%
patient_id = "Patient ID"
label = "label"
n_classes=24

if n_classes > 2:
    subtyping=True
else:
    subtyping=False
    
# %%

embedding_weights = r"C:\Users\omnia\OneDrive - University of Jeddah\PhD progress\DNA_methyalation\src\mil_vectors"  + "/embedding_" + "_" + label + ".pth"
classification_weights = r"C:\Users\omnia\OneDrive - University of Jeddah\PhD progress\DNA_methyalation\src\mil_vectors"  + "/classification_"  + "_" + label + ".pth"

# %%

df = df.dropna(subset=[label])

# %%

train_transform = transforms.Compose([
        transforms.Resize((224, 224)),                            
        #transforms.ColorJitter(brightness=0.005, contrast=0.005, saturation=0.005, hue=0.005),
        transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.1),
        transforms.ColorJitter(contrast=0.1), 
        transforms.ColorJitter(saturation=0.1),
        transforms.ColorJitter(hue=0.1)]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))      
    ])

test_transform = transforms.Compose([
        transforms.Resize((224, 224)),                            
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))      
    ])

# %%

df_train, df_test, train_sub, test_sub, file_ids, train_ids, test_ids = Loaders().df_loader(df, train_transform, test_transform, train_fraction, random_state, patient_id=patient_id, label=label, subset=subset)

# %%

# weights for minority oversampling 
count = Counter(df_train.labels)
class_count = np.array(list(count.values()))
weight = 1 / class_count
samples_weight = np.array([weight[t] for t in df_train.labels])
samples_weight = torch.from_numpy(samples_weight)
sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

# %%

train_loader, test_loader = Loaders().patches_dataloader(df_train, df_test, sampler, train_batch, test_batch, num_workers, shuffle, drop_last, Loaders.collate_fn)

# %%

train_loaded_subsets, test_loaded_subsets = Loaders().slides_dataloader(train_sub, test_sub, train_ids, test_ids, train_transform, test_transform, slide_batch, num_workers, shuffle, label=label, patient_id=patient_id)      

# %%

if train_patches:
    
    embedding_net = models.vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
                    
    # Freeze training for all layers
    for param in embedding_net.parameters():
        param.require_grad = False
    
    # Newly created modules have require_grad=True by default
    num_features = embedding_net.classifier[6].in_features
    features = list(embedding_net.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, embedding_vector_size)])
    features.extend([nn.Dropout(0.5)])
    features.extend([nn.Linear(embedding_vector_size, n_classes)]) # Add our layer with n outputs
    embedding_net.classifier = nn.Sequential(*features) # Replace the model classifier

    # if use_gpu:
    #     embedding_net.cuda() 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(embedding_net.parameters(), lr=0.0001, momentum=0.9)

# %%

if train_patches:
    
    model = train_embedding(embedding_net, train_loader, test_loader, criterion, optimizer, num_epochs=1)
    torch.save(model.state_dict(), embedding_weights)

# %%

if train_slides:
    
    embedding_net = VGG_embedding(embedding_weights, embedding_vector_size=embedding_vector_size, n_classes=n_classes)
    classification_net = GatedAttention(n_classes=n_classes, subtyping=subtyping) # add classification weight variable. 
    
    if use_gpu:
        embedding_net.cuda()
        classification_net.cuda()
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(classification_net.parameters(), lr=0.0001)

# %%
    
if train_slides:
    
    embedding_model, classification_model = train_att_slides(embedding_net, classification_net, train_loaded_subsets, test_loaded_subsets, loss_fn, optimizer_ft, n_classes=n_classes, bag_weight=0.7, num_epochs=1)
    torch.save(classification_model.state_dict(), classification_weights)

# %%

if testing_slides:
    
    loss_fn = nn.CrossEntropyLoss()
    
    embedding_net = VGG_embedding(embedding_weights, embedding_vector_size=embedding_vector_size, n_classes=n_classes)
    classification_net = GatedAttention(n_classes=n_classes, subtyping=subtyping)

    classification_net.load_state_dict(torch.load(classification_weights), strict=True)
    
    if use_gpu:
        embedding_net.cuda()
        classification_net.cuda()

# %%

if testing_slides:
    
    test_error, test_auc, test_accuracy, test_acc_logger, labels, prob, clsf_report, conf_matrix, sensitivity, specificity, incorrect_preds =       test_slides(embedding_net, classification_net, test_loaded_subsets, loss_fn, n_classes=2)

# %%

target_names=["Fibroid", "M/Lymphoid"]

auc_plot(labels, prob[:, 1], test_auc)
pr_plot(labels, prob[:, 1], sensitivity, specificity)
plot_confusion_matrix(conf_matrix, target_names, title='Confusion matrix', cmap=None, normalize=True)


###############################
# %%

history = soft_vote(embedding_net, test_loaded_subsets)

# %%
