import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


'''
Reference: 'Single-side domain generalization for face anti-spoofing' (CVPR'20)
- https://arxiv.org/abs/2004.14043
'''


def softplus(x):
    return torch.nn.functional.softplus(x, beta=100)


class FWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = 128
        self.gamma1 = torch.nn.Parameter(torch.ones(1, self.c1, 1, 1) * 0.3)
        self.beta1 = torch.nn.Parameter(torch.ones(1, self.c1, 1, 1) * 0.5)
        self.conv = nn.Sequential(
            nn.Conv2d(self.c1, self.c1, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.c1, self.c1, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.c1, self.c1, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.training:
            x = self.conv(x)
            x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
            x = torch.div(x, x_norm)
            gamma = (1 + torch.randn(1, self.c1, 1, 1, device=self.gamma1.device) * softplus(self.gamma1)).expand_as(x)
            beta = (torch.randn(1, self.c1, 1, 1, device=self.beta1.device) * softplus(self.beta1)).expand_as(x)
            out = gamma * x + beta
            return out


class conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        out_normal = self.conv(x)
        return out_normal


class LearnableAdaIN(nn.Module):
    def __init__(self):
        super().__init__()
        # self.sigma_y = nn.Parameter(torch.randn([128]), requires_grad=True)
        # self.mu_y = nn.Parameter(torch.randn([128]), requires_grad=True)

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x, (2, 3)) / (x.shape[2] * x.shape[3])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt(
            (torch.sum((x.permute([2, 3, 0, 1]) - self.mu(x)).permute([2, 3, 0, 1]) ** 2,
                       (2, 3)) + 0.000000023) / (x.shape[2] * x.shape[3]))

    def forward(self, x, sigma_y, mu_y):
        # sigma_y = torch.randn(x.size()[:2]).cuda()
        # mu_y = torch.randn(x.size()[:2]).cuda()
        # # sigma_y==>[sigma_y,sigma_y,sigma_y] based on batch size
        # for i in range(0, x.size(0)):
        #     sigma_y[i] = self.sigma_y
        #     mu_y[i] = self.mu_y
        return (sigma_y * ((x.permute([2, 3, 0, 1]) - self.mu(x)) /
                           self.sigma(x)) + mu_y).permute([2, 3, 0, 1])
    



class Ad_LDCNet(nn.Module):
    def __init__(self):
        super(Ad_LDCNet, self).__init__()
        self.Backbone = FE_Res18_learnable()
        self.LnessCsfier = Disentangled_Classifier(classes=1)
        self.DmainCsfier = Disentangled_Classifier(classes=1)
        self.FeDecoder = Feature_Decoder()
        self.LnAdaIN = LearnableAdaIN()
        self.FWT = FWT()  # Affine Feature Transform (AFT)
        self.DisCsfier = Features_Classifier(classes=1)
        self.Gtr = ConditionalGenerator()
        self.Memorybank = torch.empty((1,128,32,32)).cuda()

    def forward(self, fg, bg=None, update_step="Learn_Original"):
        if self.training:
            if update_step == "Learn_Original":
                # fixed adain
                self.Backbone.requires_grad = True
                self.LnessCsfier.requires_grad = True
                self.DmainCsfier.requires_grad = True
                self.FeDecoder.requires_grad = True
                self.LnAdaIN.requires_grad = False
                self.FWT.requires_grad = False
                self.DisCsfier.requires_grad = True
                self.Gtr.requires_grad = False
                
                # original_fg
                catfeat_fg = self.Backbone(fg)
                # disentangled feature & prediction
                f_liveness_fg, p_liveness_fg = self.LnessCsfier(catfeat_fg)
                f_liveness_norm_fg = torch.norm(f_liveness_fg, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
                f_liveness_fg = torch.div(f_liveness_fg, f_liveness_norm_fg)

                f_domain_fg, p_domain_fg = self.DmainCsfier(catfeat_fg)
                f_domain_norm_fg = torch.norm(f_domain_fg, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
                f_domain_fg = torch.div(f_domain_fg, f_domain_norm_fg)

                # original_bg
                catfeat_bg = self.Backbone(bg)
                # disentangled feature & prediction

                f_domain_bg, p_domain_bg = self.DmainCsfier(catfeat_bg)
                f_domain_norm_bg = torch.norm(f_domain_bg, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
                f_domain_bg = torch.div(f_domain_bg, f_domain_norm_bg)
                
                # randomly select fdomain-fg or fdomain-bg to form f_domain what have same length as f_liveness

                # liveness/domain feature classifier
                ########################################################################################################
                Cls     = torch.cat([f_liveness_fg, f_domain_fg, f_domain_bg], 0)
                labels  = torch.cat([torch.ones(f_liveness_fg.size(0), dtype=torch.float), torch.zeros(f_domain_fg.size(0), dtype=torch.float), torch.zeros(f_domain_bg.size(0), dtype=torch.float)], 0).cuda()
                
                indices = torch.randperm(labels.size()[0])                  
                Cls     = Cls[indices]                  
                labels  = labels[indices]                   
                    
                p_Cls = self.DisCsfier(Cls)#1 liveness, 0 domain    
                ########################################################################################################
                
                
                
                
                # reconstruct original feature
                re_catfeat = self.FeDecoder(torch.cat([f_liveness_fg, f_domain_fg], 1))   
                
                
                
                # conditional generator
                ########################################################################################################
                noise = torch.randn(1,128).cuda()
                sigma_y_con = torch.ones(1,1).cuda()
                mu_y_con = torch.zeros(1,1).cuda()
                
                sigma_y = torch.cat([sigma_y_con, noise], 1)
                mu_y = torch.cat([mu_y_con, noise], 1)
                
                sigma_y = self.Gtr(sigma_y).squeeze()
                mu_y = self.Gtr(mu_y).squeeze()
                ########################################################################################################

                # unseen domain
                f_domain_fg_hard = self.LnAdaIN(f_domain_fg, sigma_y, mu_y)
                f_domain_fg_hard_norm = torch.norm(f_domain_fg_hard, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
                f_domain_fg_hard = torch.div(f_domain_fg_hard, f_domain_fg_hard_norm)

                # reconstruct hard feature with unseen domain
                re_catfeat_hard = self.FeDecoder(torch.cat([f_liveness_fg, f_domain_fg_hard], 1))
                
                f_liveness_fg_hard, p_liveness_fg_hard = self.LnessCsfier(re_catfeat_hard)

                # diverse features
                f_liveness_fg_fwt = self.FWT(f_liveness_fg)
                f_liveness_fg_fwt_norm = torch.norm(f_liveness_fg_fwt, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (
                    2) ** 0.5
                f_liveness_fg_fwt1 = torch.div(f_liveness_fg_fwt, f_liveness_fg_fwt_norm)

                # reconstruct diverse feature with seen domain
                re_catfeat_fg_fwt = self.FeDecoder(torch.cat([f_liveness_fg_fwt1, f_domain_fg], 1))
                
                f_liveness_fg_fwt, p_liveness_fg_fwt = self.LnessCsfier(re_catfeat_fg_fwt)

                return catfeat_fg, p_liveness_fg, f_liveness_fg, f_domain_fg, f_domain_bg, re_catfeat, \
                       p_liveness_fg_fwt, \
                       p_liveness_fg_hard,\
                       p_Cls, labels, f_liveness_fg_fwt1, p_domain_fg, p_domain_bg
            elif update_step == "Learn_FWT":
                # update Affine Feature Transform (AFT)
                self.Backbone.requires_grad = False
                self.LnessCsfier.requires_grad = False
                self.DmainCsfier.requires_grad = False
                self.FeDecoder.requires_grad = False
                self.LnAdaIN.requires_grad = False
                self.FWT.requires_grad = True
                self.DisCsfier.requires_grad = False
                self.Gtr.requires_grad = False


                catfeat_fg = self.Backbone(fg)
                # disentangled feature & prediction
                f_liveness_fg, p_liveness_fg = self.LnessCsfier(catfeat_fg)
                f_liveness_fg_norm = torch.norm(f_liveness_fg, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
                f_liveness_fg = torch.div(f_liveness_fg, f_liveness_fg_norm)

                f_domain_fg, p_domain_fg = self.DmainCsfier(catfeat_fg)
                f_domain_fg_norm = torch.norm(f_domain_fg, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
                f_domain_fg = torch.div(f_domain_fg, f_domain_fg_norm)

                # diverse features
                f_liveness_fg_fwt = self.FWT(f_liveness_fg)
                f_liveness_fg_fwt_norm = torch.norm(f_liveness_fg_fwt, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (
                    2) ** 0.5
                f_liveness_fg_fwt = torch.div(f_liveness_fg_fwt, f_liveness_fg_fwt_norm)
                
                # if len(self.Memorybank) < 20:
                #     self.Memorybank = torch.cat([self.Memorybank, torch.mean(f_liveness_fg_fwt,0).unsqueeze(0)], 0)
                
                # for i in range(len(self.Memorybank)):
                #     sim = torch.abs(torch.mean(F.cosine_similarity((torch.mean(f_liveness_fg_fwt,0)), (self.Memorybank[i]), dim=0)))
                #     if sim <0.3:
                #         self.Memorybank = torch.cat([self.Memorybank[1:], (torch.mean(f_liveness_fg_fwt,0)).unsqueeze(0)], 0)
                #         continue
                
                # if memorybank is empty, add first item
                for i in range(len(f_liveness_fg_fwt)):
                    if len(self.Memorybank) < 10:
                        self.Memorybank = torch.cat([self.Memorybank, f_liveness_fg_fwt[i].unsqueeze(0)], 0)
                        # if there is any nan in memorybank, remove the nan
                        if torch.isnan(self.Memorybank).any():
                            nan_mask = torch.isnan(self.Memorybank).view(self.Memorybank.size(0), -1).any(dim=1)
                            self.Memorybank = self.Memorybank[~nan_mask]
                    else:
                        for j in range(len(self.Memorybank)):
                            # print(torch.mean(f_liveness_fg_fwt,0).shape)
                            sim = torch.abs(torch.mean(F.cosine_similarity((f_liveness_fg_fwt[i]), (self.Memorybank[j]), dim=0)))
                            if sim < 0.3:
                                # remove first item in memorybank and add new item
                                self.Memorybank = torch.cat([self.Memorybank[1:], f_liveness_fg_fwt[i].unsqueeze(0)], 0)
                                break
                
                # reconstruct diverse feature with seen domain
                re_catfeat_fwt = self.FeDecoder(torch.cat([f_liveness_fg_fwt, f_domain_fg], 1))

                # diverse feature disentanglement
                f_liveness_fg_fwt, p_liveness_fg_fwt = self.LnessCsfier(re_catfeat_fwt)
                f_liveness_fg_fwt_norm = torch.norm(f_liveness_fg_fwt, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (
                    2) ** 0.5
                f_liveness_fg_fwt = torch.div(f_liveness_fg_fwt, f_liveness_fg_fwt_norm)

                return p_liveness_fg_fwt, f_liveness_fg, f_liveness_fg_fwt, self.Memorybank

            elif update_step == "Learn_Adain":
                # update Learnable adaIN
                self.Backbone.requires_grad = False
                self.LnessCsfier.requires_grad = False
                self.DmainCsfier.requires_grad = False
                self.FeDecoder.requires_grad = False
                self.LnAdaIN.requires_grad = True
                self.FWT.requires_grad = False
                self.DisCsfier.requires_grad = False
                self.Gtr.requires_grad = True


                catfeat_fg = self.Backbone(fg)
                # disentangled feature & prediction
                f_liveness_fg, p_liveness_fg = self.LnessCsfier(catfeat_fg)
                f_domain_fg, p_domain_fg = self.DmainCsfier(catfeat_fg)

                f_domain_fg_norm = torch.norm(f_domain_fg, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
                f_domain_fg = torch.div(f_domain_fg, f_domain_fg_norm)

                # conditional generator
                ########################################################################################################
                noise = torch.randn(1,128).cuda()
                sigma_y_con = torch.ones(1,1).cuda()
                mu_y_con = torch.zeros(1,1).cuda()
                
                sigma_y = torch.cat([sigma_y_con, noise], 1)
                mu_y = torch.cat([mu_y_con, noise], 1)
                
                sigma_y = self.Gtr(sigma_y)
                mu_y = self.Gtr(mu_y)
                ########################################################################################################

                # unseen domain
                f_domain_fg_hard = self.LnAdaIN(f_domain_fg,sigma_y,mu_y)
                f_domain_fg_hard_norm = torch.norm(f_domain_fg_hard, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (
                    2) ** 0.5
                f_domain_fg_hard = torch.div(f_domain_fg_hard, f_domain_fg_hard_norm)

                re_catfeat_adain = self.FeDecoder(torch.cat([f_liveness_fg, f_domain_fg_hard], 1))
                
                f_liveness_hard, p_liveness_fg_hard = self.LnessCsfier(re_catfeat_adain)

                p_Cls = self.DisCsfier(f_domain_fg_hard)
                return  p_liveness_fg_hard, p_Cls
                
        else:
            if update_step == 'feature_map_test':
                # original_fg
                catfeat_fg = self.Backbone(fg)
                # disentangled feature & prediction
                f_liveness_fg, p_liveness_fg = self.LnessCsfier(catfeat_fg)
                f_liveness_norm_fg = torch.norm(f_liveness_fg, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
                f_liveness_fg = torch.div(f_liveness_fg, f_liveness_norm_fg)

                f_domain_fg, p_domain_fg = self.DmainCsfier(catfeat_fg)
                f_domain_norm_fg = torch.norm(f_domain_fg, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
                f_domain_fg = torch.div(f_domain_fg, f_domain_norm_fg)

                
                re_catfeat = self.FeDecoder(torch.cat([f_liveness_fg, f_domain_fg], 1))   
                return catfeat_fg, re_catfeat
            else:
                catfeat = self.Backbone(fg)
                f_liveness, p_liveness = self.LnessCsfier(catfeat)
                return p_liveness


class FE_Res18_learnable(nn.Module):
    def __init__(self):
        super(FE_Res18_learnable, self).__init__()

        model_resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3

    def forward(self, input):
        feature = self.conv1(input)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        feature1 = self.layer1(feature)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        re_feature1 = F.adaptive_avg_pool2d(feature1, 32)
        re_feature2 = F.adaptive_avg_pool2d(feature2, 32)
        re_feature3 = F.adaptive_avg_pool2d(feature3, 32)
        catfeat = torch.cat([re_feature1, re_feature2, re_feature3], 1)
        # L2 normalize
        feature_norm = torch.norm(catfeat, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
        catfeat = torch.div(catfeat, feature_norm)

        return catfeat


class Feature_Decoder(nn.Module):
    def __init__(self, in_channels=256, out_channels=448):
        super(Feature_Decoder, self).__init__()
        self.feature_decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=3, padding=1),
        )

    def forward(self, x):
        x1 = self.feature_decoder(x)
        return x1


class Disentangled_Classifier(nn.Module):
    def __init__(self, in_channels=448, classes=2, conv3x3=conv3x3):
        super(Disentangled_Classifier, self).__init__()
        self.conv1 = nn.Sequential(
            conv3x3(in_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            conv3x3(128, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier_layer = nn.Linear(512, classes)
        
        self.classes = classes

    def forward(self, x):
        x1 = self.conv1(x)
        if self.classes == 0:
            return x1, None
        x = self.conv2(x1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        classification = self.classifier_layer(x)

        return x1, classification


class Features_Classifier(nn.Module):
    def __init__(self, in_channels=128, classes=2, conv3x3=conv3x3):
        super(Features_Classifier, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier_layer = nn.Linear(128, classes)
        
        self.classes = classes

    def forward(self, x):

        x = self.avgpool(x) # 1 liveness, 0 domain
        x = x.view(x.size(0), -1)
        classification = self.classifier_layer(x)

        return classification

# class Features_Classifier(nn.Module):
#     def __init__(self, in_channels=128, classes=2):
#         super(Features_Classifier, self).__init__()

#         self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)  # Learnable conv instead of avgpool
#         self.bn = nn.BatchNorm2d(in_channels)  # Optional: normalize feature maps
#         self.relu = nn.ReLU(inplace=True)  # Optional: introduce non-linearity
#         self.classifier_layer = nn.Linear(in_channels, classes)  # Fully connected layer

#     def forward(self, x):
#         x = self.conv(x)  # Convolutional layer
#         x = self.bn(x)  # Normalize (if used)
#         x = self.relu(x)  # Activation function (if used)

#         x = torch.mean(x, dim=(2, 3))  # Global average pooling to reduce spatial dimensions
#         classification = self.classifier_layer(x)

#         return classification
    
class ConditionalGenerator(nn.Module):
    def __init__(self):
        super(ConditionalGenerator, self).__init__()
        
        # Define the generator architecture
        self.net = nn.Sequential(
            # nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1),
            # # nn.ReLU(),
            # nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            # # nn.ReLU(),
            # nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            # # nn.AdaptiveAvgPool2d((1, 1)),
            # nn.Flatten(),
            # nn.Linear(1024, 128),
            nn.Linear(129, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            
        )
        

    def forward(self, x):
        x =self.net(x)
        
        return x

if __name__ == "__main__":
    net = Ad_LDCNet().cuda()
    net(torch.randn(2, 3, 256, 256).cuda())
