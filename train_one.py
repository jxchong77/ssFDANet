from models.DFA_one import *
import torch.optim as optim
import numpy as np
import os
import random
import torchvision.transforms as T
import torch.functional as F
from losses import *
import logging
from pytz import timezone
from datetime import datetime
import sys

logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def get_data_loader(fg=[], bg=[], batch_size=5, shuffle=True, drop_last=True):

    data_fg = None
    data_bg = None
    
    for i in range(len(fg)):
        temp_fg = np.load(fg[i])
        temp_bg = np.load(bg[i])
        print('temp_fg.shape={} temp_bg.shape={}'.format(temp_fg.shape, temp_bg.shape))

        if len(temp_bg) < len(temp_fg):
            for j in range(int(len(temp_fg)/len(temp_bg))):
                temp_bg = np.concatenate((temp_bg, temp_bg), axis=0)
            temp_bg = temp_bg[:len(temp_fg)]
        print('temp_fg.shape={} temp_bg.shape={}'.format(temp_fg.shape, temp_bg.shape))

        if i > 0:
            data_fg =  np.concatenate((data_fg, temp_fg), axis=0)
            data_bg =  np.concatenate((data_bg, temp_bg), axis=0)
        else:
            data_fg = temp_fg#[:length]
            data_bg = temp_bg#[:length]
    
    print('data_fg.shape={} data_bg.shape={}'.format(data_fg.shape, data_bg.shape))

    # dataset
    trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(data_fg, (0, 3, 1, 2))),
                                              torch.tensor(np.transpose(data_bg, (0, 3, 1, 2))))
    # free memory
    import gc
    del data_fg, data_bg
    gc.collect()
    # dataloader
    data_loader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              drop_last=drop_last)
    return data_loader


def get_inf_iterator(data_loader):
    # """Inf data iterator."""
    while True:
        for images_fg, images_bg in data_loader:
            yield (images_fg, images_bg)


def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))


dataset1 = "MSU"
dataset2 = "replay"


device_id = "cuda:0" 
results_path = f"/shared/Jxchong/OS_{dataset1}_{dataset2}/"

if os.path.exists(results_path):
    import shutil
    shutil.rmtree(results_path)

mkdir(results_path)




batch_size = 4
log_step = 20
model_save_step = 1
model_save_epoch = 1
save_index = 0

live_path1fg =    '/shared/shared/domain-generalization/' + dataset1 + '_images_live_fg1.npy'
live_path1bg =    '/shared/shared/domain-generalization/' + dataset1 + '_images_live_bg1.npy'

live_path2fg =    '/shared/shared/domain-generalization/' + dataset2 + '_images_live_fg1.npy'
live_path2bg =    '/shared/shared/domain-generalization/' + dataset2 + '_images_live_bg1.npy'


Fas_Net = Ad_LDCNet().to(device_id)
Bcriterion_ce = nn.BCEWithLogitsLoss().to(device_id)
criterionMSE = torch.nn.MSELoss().to(device_id)
criterion_cosine = nn.CosineSimilarity().to(device_id)

num_epochs = 5
lr = 5e-3

step_size = 5
gamma = 0.8

betasA = 0.3
betasB = 0.5

optimizers = optim.NAdam

optimizer_fas =     optimizers(Fas_Net.parameters(), lr=lr, betas=(betasA, betasB))
optimizer_fwt =     optimizers(Fas_Net.parameters(), lr=lr, betas=(betasA, betasB))
optimizer_adain =   optimizers(Fas_Net.parameters(), lr=lr, betas=(betasA, betasB))

scheduler_fas = torch.optim.lr_scheduler.StepLR(optimizer_fas, step_size=step_size, gamma=gamma)
scheduler_fwt = torch.optim.lr_scheduler.StepLR(optimizer_fwt, step_size=step_size, gamma=gamma)
scheduler_adain = torch.optim.lr_scheduler.StepLR(optimizer_adain, step_size=step_size, gamma=gamma)

Fas_Net.train()


data1_real = get_data_loader(fg=[live_path1fg,live_path2fg], bg=[live_path1bg,live_path2bg],
                             batch_size=batch_size, shuffle=True)

iternum = len(data1_real)


print('iternum={}'.format(iternum))
data1_real = get_inf_iterator(data1_real)



T_transform = torch.nn.Sequential(
        T.Pad(40, padding_mode="symmetric"),
        T.RandomRotation(30),
        T.RandomHorizontalFlip(p=0.5),
        T.Resize(286),
        T.RandomCrop(256),
        )

RandomCrop = torch.nn.Sequential(
        T.RandomCrop(size=(128,128)) 
)


results_filename = os.path.split(results_path)[-1]
file_handler = logging.FileHandler(filename='/home/Jxchong/icme_ext/logger/train/'+results_filename+'.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
date = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)

logging.info('lr={}'.format(lr))
logging.info('batch_size={}'.format(batch_size))
logging.info('betasA={} betasB={}'.format(betasA, betasB))
logging.info('optimizers={}'.format(optimizers))

for epoch in range(num_epochs):

    for step in range(iternum):
        # ============ one batch extraction ============#
        imgs1_fg, imgs1_bg = next(data1_real)

        imgs_fg = imgs1_fg.to(device_id).float()
        imgs_bg = imgs1_bg.to(device_id).float()
        
        
        batchidx = list(range(len(imgs_fg)))
        random.shuffle(batchidx)

        img_rand_fg = imgs_fg[batchidx, :]
        img_rand_bg = imgs_bg[batchidx, :]

        img_rand_fg = NormalizeData_torch(T_transform(RandomCrop(img_rand_fg)))
        img_rand_bg = NormalizeData_torch(T_transform(img_rand_bg))
####################################################################
        # Learn_Original
        catfeat_fg, p_liveness_fg, f_liveness_fg, f_domain_fg, f_domain_bg, re_catfeat, \
        p_liveness_fg_fwt, \
        p_liveness_fg_hard,\
        p_Cls, Cls_labels, f_liveness_fg_fwt1, _, _ = \
            Fas_Net(img_rand_fg, img_rand_bg, update_step="Learn_Original")

        # self supervised loss
        f_domain_fg = F.normalize(f_domain_fg, p=2, dim=1)
        f_domain_bg = F.normalize(f_domain_bg, p=2, dim=1)
        f_liveness_fg = F.normalize(f_liveness_fg, p=2, dim=1)

        Loss_dis = dissimilar_cosine(torch.cat((f_domain_fg, f_domain_bg), 0), f_liveness_fg)

        Loss_d_sim = 1 - torch.mean(criterion_cosine(f_domain_fg, f_domain_bg))

        Loss_f_sim = similar_cosine(f_liveness_fg)

        # reconstruction loss
        Loss_re = criterionMSE(catfeat_fg, re_catfeat)
        
        Loss_Dis_Cls = Bcriterion_ce(p_Cls.squeeze(), Cls_labels)

        Loss_ls = Bcriterion_ce(p_liveness_fg.squeeze(), torch.ones(len(p_liveness_fg), dtype=torch.float).cuda())
        Loss_ls += Bcriterion_ce(p_liveness_fg_fwt.squeeze(), torch.zeros(len(p_liveness_fg_fwt), dtype=torch.float).cuda())

        Loss_ls_hard = Bcriterion_ce(p_liveness_fg_hard.squeeze(), torch.ones(len(p_liveness_fg_hard), dtype=torch.float).cuda())

        Loss_fwt_MSE = criterionMSE(f_liveness_fg_fwt1, f_liveness_fg)


        Loss_all = Loss_ls + Loss_dis + Loss_d_sim + Loss_re + Loss_f_sim + \
            Loss_Dis_Cls + Loss_fwt_MSE + Loss_ls_hard
        # if lossall is nan break
        if torch.isnan(Loss_all):
            print('Loss_all is nan, break')
            break
        optimizer_fas.zero_grad()
        Loss_all.backward()
        optimizer_fas.step()

        if (step + 1) % log_step == 0:
            logging.info('[epoch %d step %d] Fixed_ FWT Loss_ls %.4f   Loss_dis %.4f Loss_d_sim %.4f Loss_f_sim %.4f'
                  ' Loss_Dis_Cls %.4f Loss_re %.4f Loss_fwt_MSE %.4f   Loss_ls_hard %.4f'
                  % (epoch, step, Loss_ls.item(), Loss_dis.item(), Loss_d_sim.item(),
                     Loss_f_sim.item(),
                     Loss_Dis_Cls.item(), Loss_re.item(), Loss_fwt_MSE.item(), Loss_ls_hard.item()))
####################################################################
        # Learn_AFT
        p_liveness_fwt, \
        f_liveness, f_liveness_fwt, Memorybank \
            = Fas_Net(img_rand_fg, img_rand_bg, update_step="Learn_FWT")

        # fwt sample loss
        Loss_ls_fwt = Bcriterion_ce(p_liveness_fwt.squeeze(), torch.zeros(len(p_liveness_fwt), dtype=torch.float).cuda())
        

        # fwt sample and live sample similarity (encourage the dissimilarity)
        f_liveness = F.normalize(f_liveness, p=2, dim=1)
        f_liveness_fwt = F.normalize(f_liveness_fwt, p=2, dim=1)
        Memorybank = F.normalize(Memorybank, p=2, dim=1).detach()

        Loss_dissimilar = torch.mean(torch.abs(criterion_cosine(f_liveness, f_liveness_fwt)))

        # mask the feature map channel wise
        mask = torch.rand(f_liveness_fwt.shape[0], f_liveness_fwt.shape[1], 1, 1).to(device_id)
        mask = (mask > 0.5).float()
        f_liveness_fwt_masked = f_liveness_fwt * mask

        Loss_mine = contrastive_loss(f_liveness_fwt,f_liveness_fwt_masked,Memorybank)


        Loss_all_fwt = Loss_dissimilar + Loss_ls_fwt + 0.1*Loss_mine
        optimizer_fwt.zero_grad()
        Loss_all_fwt.backward()
        optimizer_fwt.step()
        
        if (step + 1) % log_step == 0:
            logging.info('[epoch %d step %d] Update FWT ,Loss_ls_fwt %.4f ,Loss_ls_dis %.4f  Loss_mine %.4f'
                  % (epoch, step,  Loss_ls_fwt.item(), Loss_dissimilar.item() , Loss_mine.item()))
####################################################################
        # Learn_Adain
        p_liveness_hard, p_Cls \
            = Fas_Net(img_rand_fg, img_rand_bg, update_step="Learn_Adain")
  
        # ladain sample loss with (1 - ls_lab_rand) (encourage strong domain augmentation)
        Loss_ls_hard = Bcriterion_ce(p_liveness_hard.squeeze(), 1 - torch.ones(len(p_liveness_hard), dtype=torch.float).cuda())

        Loss_Dis_Cls = Bcriterion_ce(p_Cls.squeeze(), torch.zeros(len(p_Cls), dtype=torch.float).cuda())
        
        Loss_all_adain = Loss_ls_hard + Loss_Dis_Cls
        optimizer_adain.zero_grad()
        Loss_all_adain.backward()
        optimizer_adain.step()
        if (step + 1) % log_step == 0:
            logging.info('[epoch %d step %d] Update LnAdaIN ,Loss_ls_hard %.4f  Loss_Dis_Cls %.4f'
                  % (epoch, step, Loss_ls_hard.item(), Loss_Dis_Cls.item()))
####################################################################

        if ((step + 1) % model_save_step == 0):
            mkdir(results_path)
            save_index += 1
            torch.save(Fas_Net.state_dict(), os.path.join(results_path,
                                                          "FASNet-{}.tar".format(save_index)))

    if ((epoch + 1) % model_save_epoch == 0):
        mkdir(results_path)
        save_index += 1
        torch.save(Fas_Net.state_dict(), os.path.join(results_path,
                                                      "FASNet-{}.tar".format(save_index)))

    try:
        scheduler_fas.step()
        scheduler_fwt.step()
        scheduler_adain.step()
    except Exception:
        pass
    logging.info('[epoch %d] lrs: fas=%.6g fwt=%.6g adain=%.6g' % (
        epoch,
        optimizer_fas.param_groups[0]['lr'],
        optimizer_fwt.param_groups[0]['lr'],
        optimizer_adain.param_groups[0]['lr']

    ))
