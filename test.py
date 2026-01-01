
from models.DFA_one import *
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
import torch
import os
import logging
from pytz import timezone
from datetime import datetime
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def Find_Optimal_Cutoff(TPR, FPR, threshold):
    # y = TPR - FPR
    y = TPR + (1 - FPR)
    # print(y)
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def NormalizeData(data):
    if np.min(data) ==np.max(data):
        a = data
    else:
        a = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    return  a

def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

import sys
args = sys.argv[1]

test_dataset = str(args)

# live_path   = '/shared/shared/3Dmask/'+test_dataset+'_images_live.npy'
# spoof_path  = '/shared/shared/3Dmask/'+test_dataset+'_images_spoof.npy'

live_path   = '/shared/shared/domain-generalization/'+test_dataset+'_images_live.npy'
spoof_path  = '/shared/shared/domain-generalization/'+test_dataset+'_images_spoof.npy'


live_data = np.load(live_path)
spoof_data = np.load(spoof_path)
live_label = np.ones(len(live_data), dtype=np.int64)
spoof_label = np.zeros(len(spoof_data), dtype=np.int64)

total_data = np.concatenate((live_data, spoof_data), axis=0)
total_label = np.concatenate((live_label, spoof_label), axis=0)

print(live_data.shape, len(total_data))

trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(total_data, (0, 3, 1, 2))),
                                          torch.tensor(total_label))
# dataloader
data_loader = torch.utils.data.DataLoader(trainset,
                                          batch_size=1000,
                                          shuffle=False, )
device_id = "cuda:0" 
FASNet = Ad_LDCNet().to(device_id)

model_path = "/shared/Jxchong/OS_MSU_replay/FASNet-"

print("model_path", model_path)
print("live_path", live_path)
print("spoof_path", spoof_path)
 
import glob
length = len(glob.glob(model_path+"*.tar")) 

#results_filename is split of model_path on '/' and last element
results_filename = model_path.split('/')[-2]
print(results_filename)
file_handler = logging.FileHandler(filename='/home/Jxchong/icme_ext/logger/test/'+results_filename + '_' + test_dataset +'.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
date = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)


record = [1,100,100,100,100]
with torch.no_grad():
    # for epoch in reversed(range(1,length)):
    for epoch in (range(1,length)):
        FASNet_path = model_path + str(epoch) + ".tar"


        FASNet.load_state_dict(torch.load(FASNet_path, map_location=device_id),strict=False) 
        FASNet.eval()

        score_list_ori = []
        score_list_spoof = []
        Total_score_list_cs = []
        label_list = []
        TP = 0.0000001
        TN = 0.0000001
        FP = 0.0000001
        FN = 0.0000001

        for i, data in enumerate(data_loader, 0):
            images, labels = data
            images = images.to(device_id)
            label_pred = FASNet(NormalizeData_torch(images))
            
            #calculate score with the sum of all channels and pixels
            # score = torch.mean(label_pred, dim=(1,2,3)).cpu().data.numpy()

            # score = F.softmax(label_pred, dim=1).cpu().data.numpy()[:, 0]  # multi class
            score = torch.sigmoid(label_pred).cpu().data.numpy().squeeze()  # binary class
            for j in range(images.size(0)):
                score_list_ori.append(score[j]) 
                label_list.append(labels[j])


        # print(max(score_list_ori), min(score_list_ori))
        # score_list_ori = NormalizeData(score_list_ori)

        for i in range(0, len(label_list)):
            Total_score_list_cs.append(score_list_ori[i]) 
            if score_list_ori[i] == None:
                print(score_list_ori[i])
        fpr, tpr, thresholds_cs = metrics.roc_curve(label_list, Total_score_list_cs)
        threshold_cs, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds_cs)

        for i in range(len(Total_score_list_cs)):
            score = Total_score_list_cs[i]
            if (score >= threshold_cs and label_list[i] == 1):
                TP += 1
            elif (score < threshold_cs and label_list[i] == 0):
                TN += 1
            elif (score >= threshold_cs and label_list[i] == 0):
                FP += 1
            elif (score < threshold_cs and label_list[i] == 1):
                FN += 1

        APCER = FP / (TN + FP)
        NPCER = FN / (FN + TP)
        
        if record[1]>((APCER + NPCER) / 2):
                record[0]=epoch
                record[1]=((APCER + NPCER) / 2)
                record[2]=roc_auc_score(label_list, score_list_ori)
                record[3]=APCER
                record[4]=NPCER
                
        logging.info('[epoch %d]  APCER %.4f  NPCER %.4f         ACER %.4f  AUC %.4f'
                % (epoch, APCER, NPCER, np.round((APCER + NPCER) / 2, 4), np.round(roc_auc_score(label_list, score_list_ori), 4)))

    logging.info(f"BEST Epoch {str(record[0])} ACER {str(record[1])} AUC {str(record[2])} APCER {str(record[3])} NPCER {str(record[4])} ")


