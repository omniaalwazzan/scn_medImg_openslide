import os
import os.path
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter
import pickle
import argparse
import time
import gc
gc.enable()

from PIL import Image
from PIL import ImageFile

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc as calc_auc

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F

# internal import 

from loaders import Loaders
from embedding_net import VGG_embedding
from create_store_graphs import create_embeddings_graphs
from graph_train_loop import train_graph_multi_wsi, test_graph_multi_wsi
from auxiliary_functions import *

# %%

# Set environment variables
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Check for GPU availability
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
#device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set image properties
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

plt.ion()


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

# Parameters
seed = 42
#seed_everything(seed)
train_fraction = 0.70
subset= False # TODO
slide_batch = 2#10 # TODO. change Dataset to Iterable dataset to solve this problem. this needs to be larger than one, otherwise Dataloader can fail when only passed a None object from collate function.
num_workers = 0
batch_size = 2 #10
creating_embedding = True  # TODO
train_graph = True  # TODO
embedding_vector_size = 1024
bag_weight=0.7

learning_rate = 0.0001
pooling_ratio = 0.7
heads =4
num_epochs = 2
TRAIN = True
TEST = True

label = 'label'
patient_id = 'Patient ID'
dataset_name = 'RA'
n_classes= 22 #6 
#PATH_patches = "/data/DERI-MMH/DNA_meth/MIL/Tiles_data.csv" # EDIT 
PATH_patches =r"C:/Users/omnia/OneDrive - University of Jeddah/brain/proj2/Tiles_data.csv"
checkpoint = True
andrena_path = "/data/DERI-MMH/DNA_meth/MIL" # EDIT
current_directory = Path(r"C:\Users\omnia\OneDrive - University of Jeddah\PhD progress\DNA_methyalation\src\MUSTANG").resolve().parent
run_results_folder = f"graph_{dataset_name}_{seed}_{heads}_{pooling_ratio}_{learning_rate}"
results = os.path.join(current_directory, "results/" + run_results_folder)
checkpoints = results + "/checkpoints"
os.makedirs(results, exist_ok = True)
os.makedirs(checkpoints, exist_ok = True)

# Load the dataset
df = pd.read_csv(PATH_patches, header=0)
df = df.rename(columns={'ID': 'Patient ID'})
df = df.dropna(subset=[label])
df['label'].nunique()

#%%
# Define collate function
def collate_fn_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
#%%
'''
 file_id contains unique sample Ids to use for train and test split
 train and test subset are datframe contines the batches coresspoinding to uniqe train/test ids

'''
# file_ids, train_ids, test_ids = Loaders().train_test_ids(df, train_fraction, seed, patient_id, label, subset)
# train_subset, test_subset = Loaders().df_loader(df, train_transform, test_transform, train_ids, test_ids, patient_id, label, subset=subset)
# #%%
# # train_slides is 
# train_slides, test_slides = Loaders().slides_dataloader(train_subset, test_subset, train_ids, test_ids, train_transform, test_transform, slide_batch=slide_batch, num_workers=num_workers, shuffle=False, collate=collate_fn_none, label=label, patient_id=patient_id)
# embedding_net = VGG_embedding(embedding_vector_size=embedding_vector_size, n_classes=n_classes)
# embedding_net.cpu()

# #%%
# slides_dict = {'train_embedding_dict_' : train_slides , 'test_embedding_dict_': test_slides}
# for file_prefix, slides in slides_dict.items():
#     embedding_dict = create_embeddings_graphs(embedding_net, slides,include_self=False)
#     print(f"Started saving {file_prefix[:-18]} to file")
#     with open(f"{file_prefix[1]}{dataset_name}.pkl", "wb") as file:
#         pickle.dump(embedding_dict, file)  # encode dict into Pickle
#         print("Done writing embedding dict into pickle file")
#%%
    # create patch embeddings with VGG16_bn
if creating_embedding:
    file_ids, train_ids, test_ids = Loaders().train_test_ids(df, train_fraction, seed, patient_id, label, subset)
    train_subset, test_subset = Loaders().df_loader(df, train_transform, test_transform, train_ids, test_ids, patient_id, label, subset=subset)
    train_slides, test_slides = Loaders().slides_dataloader(train_subset, test_subset, train_ids, test_ids, train_transform, test_transform, slide_batch=slide_batch, num_workers=num_workers, shuffle=False, collate=collate_fn_none, label=label, patient_id=patient_id)
    embedding_net = VGG_embedding(embedding_vector_size=embedding_vector_size, n_classes=n_classes)
    if not use_gpu:
        embedding_net.cuda()
    # Save k-NNG with VGG patch embedddings for future use
    slides_dict = {'train_embedding_dict_' : train_slides , 'test_embedding_dict_': test_slides}
    for file_prefix, slides in slides_dict.items():
        embedding_dict = create_embeddings_graphs(embedding_net, slides,include_self=False)
        print(f"Started saving {file_prefix[:-18]} to file")
        with open(f"{file_prefix[:-17]}{dataset_name}.pkl", "wb") as file:
            pickle.dump(embedding_dict, file)  # encode dict into Pickle
            print("Done writing embedding dict into pickle file")
            
# %%\
import os
creating_embedding = False  # TODO
if not creating_embedding:
    with open(f"r{dataset_name}.pkl", "rb") as train_file:
    # Load the dictionary from the file
        train_embedding_dict = pickle.load(train_file)
    with open(f"e{dataset_name}.pkl", "rb") as test_file:
    # Load the dictionary from the file
        test_embedding_dict = pickle.load(test_file)

#%%
# %%
train_loader = torch.utils.data.DataLoader(train_embedding_dict, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False)
#train_graph_loader = torch_geometric.loader.DataLoader(train_graph_dict, batch_size=1, shuffle=False, num_workers=0, sampler=sampler, drop_last=False, generator=seed_everything(state)) #TODO MINIBATCHING
test_loader = torch.utils.data.DataLoader(test_embedding_dict, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

#%%


class GatedAttention(nn.Module):
        
    """
    L: input feature dimension
    D: hidden layer dimension
    Dropout: True or False
    n_classes: number of classes
    """
    
    def __init__(self, L= 1024, D=224, Dropout=True, n_classes = 6, k_sample=8, instance_loss_fn=nn.CrossEntropyLoss(), subtyping=True):
        
        super(GatedAttention, self).__init__()
        self.L = L
        self.D = D
        self.Dropout= Dropout
        self.n_classes = n_classes
        self.instance_loss_fn = instance_loss_fn
        self.subtyping = subtyping
        self.k_sample = k_sample
       
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        
        if self.Dropout:
            self.attention_V.append(nn.Dropout(0.25))
            self.attention_U.append(nn.Dropout(0.25))
            
        self.attention_V = nn.Sequential(*self.attention_V)
        self.attention_U = nn.Sequential(*self.attention_U)
            
        self.attention_weights = nn.Linear(self.D, 1) 

        self.classifier = nn.Sequential(
            nn.Linear(self.L, self.n_classes)
        ) 

        instance_classifiers = [nn.Linear(self.L, 2) for i in range(n_classes)] #  n_classes?
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        
    def forward(self, x, label =None,instance_eval=True):
        #print("x shape is\n",x.shape)
        #print("label",label)

        
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK ##################
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        
        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label.long(), num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, x, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, x, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, x)  # KxL

        logits = self.classifier(M) #logits
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        #Y_hat = torch.ge(Y_prob, 0.5).float()
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}

        return logits, Y_prob, Y_hat, A, results_dict

    # AUXILIARY METHODS
    def calculate_error(self, Y_hat, Y):
    	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
    
    	return error
        
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()

    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, x, classifier):  # h=x
        device=x.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(x, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(x, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, x, classifier):
        device=x.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(x, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

# %%

subtyping= True
classification_net = GatedAttention(n_classes=n_classes, subtyping=subtyping)
loss_fn = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(classification_net.parameters(), lr=0.0001)


#%%
def train_clam_multi_wsi(clam_net, train_loader, test_loader, loss_fn, optimizer, n_classes, bag_weight, num_epochs, training=True, testing=True, checkpoint=True, checkpoint_path="PATH_checkpoints"):

    since = time.time()
    best_acc = 0.
    best_AUC = 0.
    
    results_dict= {}
    
    train_loss_list = []
    train_accuracy_list = []

    val_loss_list = []
    val_accuracy_list = []
    val_auc_list = []

    for epoch in range(num_epochs):

        ###################################
        # TRAIN

        if training:

            acc_logger = Accuracy_Logger(n_classes=n_classes)
            inst_logger = Accuracy_Logger(n_classes=n_classes)

            train_loss = 0 # train_loss
            train_error = 0
            train_inst_loss = 0.
            inst_count = 0

            train_acc = 0
            train_count = 0

            clam_net.train()

            print("Epoch {}/{}".format(epoch, num_epochs))
            print('-' * 10)

            for batch_idx, (patient_ID, embedding) in enumerate(train_loader.dataset.items()):

                data, label = embedding

                if not use_gpu:
                    data, label = data.cuda(), label.cuda()
                else:
                    data, label = data, label

                #print(patient_ID, label)
                logits, Y_prob, Y_hat, A, instance_dict = clam_net(data, label=label, instance_eval=True)
                #logits, Y_prob, Y_hat, _, instance_dict,_ = clam_net(data, label=label, instance_eval=True)
                acc_logger.log(Y_hat, label)
                loss = loss_fn(logits, label)
                #loss_value = loss.item()

                train_acc += torch.sum(Y_hat == label.data)
                train_count += 1

                instance_loss = instance_dict['instance_loss']
                inst_count+=1
                instance_loss_value = instance_loss.item()
                train_inst_loss += instance_loss_value

                total_loss = bag_weight * loss + (1-bag_weight) * instance_loss

                inst_preds = instance_dict['inst_preds']
                inst_labels = instance_dict['inst_labels']
                inst_logger.log_batch(inst_preds, inst_labels)

                #train_loss += loss_value
                train_loss += loss.item()
                error = clam_net.calculate_error(Y_hat, label)
                train_error += error

                # backward pass
                total_loss.backward()
                # step
                optimizer.step()
                optimizer.zero_grad()

            train_loss = train_loss / train_count
            train_error = train_error/ train_count
            train_accuracy =  train_acc / train_count
            
            train_loss_list.append(train_loss)
            train_accuracy_list.append(train_accuracy.item())

            if inst_count > 0:
                train_inst_loss /= inst_count
                print('\n')

            print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}, train_accuracy: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error, train_accuracy))
            for i in range(n_classes):
                acc, correct, count = acc_logger.get_summary(i)
                print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count), flush=True)

        ###################################
        # TEST

        if testing:

            clam_net.eval()

            val_acc_logger = Accuracy_Logger(n_classes)
            val_inst_logger = Accuracy_Logger(n_classes)
            val_loss = 0.
            val_error = 0.

            val_inst_loss = 0.
            val_inst_count= 0

            val_acc = 0
            val_count = 0

            prob = []
            labels = []

            for batch_idx, (patient_ID, embedding) in enumerate(test_loader.dataset.items()):

                data, label = embedding

                with torch.no_grad():
                    if not use_gpu:
                        data, label = data.cuda(), label.cuda()
                    else:
                        data, label = data, label

                #logits, Y_prob, Y_hat, _, instance_dict,_ = clam_net(data, label=label, instance_eval=True)
                logits, Y_prob, Y_hat, _, instance_dict = clam_net(data, label=label, instance_eval=True)
                val_acc_logger.log(Y_hat, label)
                val_acc += torch.sum(Y_hat == label.data)
                val_count +=1
                loss = loss_fn(logits, label)
                val_loss += loss.item()

                instance_loss = instance_dict['instance_loss']
                val_inst_count+=1
                instance_loss_value = instance_loss.item()
                val_inst_loss += instance_loss_value
                inst_preds = instance_dict['inst_preds']
                inst_labels = instance_dict['inst_labels']
                val_inst_logger.log_batch(inst_preds, inst_labels)

                prob.append(Y_prob.detach().to('cpu').numpy())
                labels.append(label.item())

                error = clam_net.calculate_error(Y_hat, label)
                val_error += error

            val_error /= val_count
            val_loss /= val_count
            val_accuracy = val_acc / val_count

            val_loss_list.append(val_loss)
            val_accuracy_list.append(val_accuracy.item())

            if n_classes == 2:
                prob =  np.stack(prob, axis=1)[0]
                val_auc = roc_auc_score(labels, prob[:, 1])
                aucs = []
            else:
                aucs = []
                binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
                prob =  np.stack(prob, axis=1)[0]
                for class_idx in range(n_classes):
                    if class_idx in labels:
                        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                        #fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[class_idx].ravel())

                        aucs.append(calc_auc(fpr, tpr))
                    else:
                        aucs.append(float('nan'))

                val_auc = np.nanmean(np.array(aucs))

            val_auc_list.append(val_auc)

            conf_matrix = confusion_matrix(labels, np.argmax(prob, axis=1))

            print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, AUC: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_error, val_auc, val_accuracy))
            if val_inst_count > 0:
                val_inst_loss /= val_inst_count

            print(conf_matrix)
            if n_classes == 2:
                sensitivity = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0]) # TP / (TP + FN)
                specificity = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
                print('Sensitivity: ', sensitivity)
                print('Specificity: ', specificity, flush=True)

            if val_accuracy >= best_acc:
                if val_auc >= best_AUC:
                    best_acc = val_accuracy
                    best_AUC = val_auc

                    if checkpoint:
                        checkpoint_weights = checkpoint_path + str(epoch) + ".pth"
                        torch.save(clam_net.state_dict(), checkpoint_weights)

    elapsed_time = time.time() - since

    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

    if checkpoint:
        clam_net.load_state_dict(torch.load(checkpoint_weights), strict=True)
        
    results_dict = {'train_loss': train_loss_list,
                    'val_loss': val_loss_list,
                    'train_accuracy': train_accuracy_list,
                    'val_accuracy': val_accuracy_list,
                    'val_auc': val_auc_list
                    }


    #return val_loss_list, val_accuracy_list, val_auc_list, clam_net
    return clam_net,results_dict



def test_clam_slides(clam_net, test_loader, loss_fn, optimizer_ft, embedding_vector_size, n_classes):

    # TEST

    since = time.time()

    test_acc_logger = Accuracy_Logger(n_classes)
    test_inst_logger = Accuracy_Logger(n_classes)
    test_loss = 0.
    test_error = 0.

    test_inst_loss = 0.
    test_inst_count= 0

    test_acc = 0
    test_count = 0

    test_loss_list = []
    test_accuracy_list = []
    test_auc_list = []

    prob = []
    labels = []

    clam_net.eval()

    for batch_idx, (patient_ID, embedding) in enumerate(test_loader.dataset.items()):

        data, label = embedding

        with torch.no_grad():
            if not use_gpu:
                data, label = data.cuda(), label.cuda()
            else:
                data, label = data, label

        #logits, Y_prob, Y_hat, _ , instance_dict,_ = clam_net(data, label=label, instance_eval=True)
        logits, Y_prob, Y_hat, _, instance_dict = clam_net(data, label=label, instance_eval=True)

        test_acc_logger.log(Y_hat, label)
        test_acc += torch.sum(Y_hat == label.data)
        test_count +=1
        loss = loss_fn(logits, label)
        test_loss += loss.item()

        instance_loss = instance_dict['instance_loss']
        test_inst_count+=1
        instance_loss_value = instance_loss.item()
        test_inst_loss += instance_loss_value
        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        test_inst_logger.log_batch(inst_preds, inst_labels)

        prob.append(Y_prob.detach().to('cpu').numpy())
        labels.append(label.item())

        error = clam_net.calculate_error(Y_hat, label)
        test_error += error

    test_error /= test_count
    test_loss /= test_count
    test_accuracy = test_acc / test_count

    test_loss_list.append(test_loss)
    test_accuracy_list.append(test_accuracy.item())

    if n_classes == 2:
        prob =  np.stack(prob, axis=1)[0]
        test_auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        prob =  np.stack(prob, axis=1)[0]
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        test_auc = np.nanmean(np.array(aucs))

    test_auc_list.append(test_auc)

    conf_matrix = confusion_matrix(labels, np.argmax(prob, axis=1))

    print('\nTesting Set, test_loss: {:.4f}, test_error: {:.4f}, AUC: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_error, test_auc, test_accuracy))
    if test_inst_count > 0:
        test_inst_loss /= test_inst_count

    print(conf_matrix)
    if n_classes == 2:
        sensitivity = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0]) # TP / (TP + FN)
        specificity = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
        print('Sensitivity: ', sensitivity)
        print('Specificity: ', specificity, flush=True)

    elapsed_time = time.time() - since

    print()
    print("Testing completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

    #return test_loss_list, test_accuracy_list, test_auc_list, labels, prob, conf_matrix, sensitivity, specificity
    return test_loss_list, test_accuracy_list, test_auc_list, labels, prob, conf_matrix


#%%
 # add classification weight variable. 

train_slides =True
if train_slides:
    #val_loss_list, val_accuracy_list, val_auc_list, clam_net= train_clam_multi_wsi(classification_net, train_loader, test_loader, loss_fn, optimizer_ft, n_classes=n_classes, bag_weight=0.7, num_epochs=20, training=True, testing=True, checkpoint=True, checkpoint_path="PATH_checkpoints")
    clam_net,results_dict= train_clam_multi_wsi(classification_net, train_loader, test_loader, loss_fn, optimizer_ft, n_classes=n_classes, bag_weight=bag_weight, num_epochs=num_epochs, training=True, testing=True, checkpoint=True, checkpoint_path="PATH_checkpoints")
    torch.save(clam_net.state_dict(), results + "\\" + run_results_folder + ".pth")
    df_results = pd.DataFrame.from_dict(results_dict)
    df_results.to_csv(results + "\\" + run_results_folder + ".csv", index=False)
    #embedding_model, classification_model = train_att_slides(embedding_net, classification_net, train_loader, test_loader, loss_fn, optimizer_ft, n_classes=n_classes, bag_weight=0.7, num_epochs=1)
    #torch.save(classification_model.state_dict(), classification_weights)
if TEST:
    test_loss_list, test_accuracy_list, test_auc_list, labels, prob, conf_matrix = test_clam_slides(clam_net, test_loader, loss_fn, optimizer_ft, embedding_vector_size, n_classes=n_classes)
