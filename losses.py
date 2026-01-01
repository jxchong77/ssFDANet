from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

def similar_cosine (features):
    Loss = 0
    # features = F.normalize(features, p=2, dim=1)
    for i in range(len(features)):
        for j in range(len(features)):
            if i != j:
                Loss += 1 - torch.mean(torch.cosine_similarity(features[i], features[j], dim=0))
    return Loss/((len(features)**2)-len(features))

def dissimilar_cosine (features1, features2):
    Loss = 0
    for i in range(len(features1)):
        for j in range(len(features2)):
            Loss += torch.mean(torch.abs(torch.cosine_similarity(features1[i], features2[j], dim=0)))
    return Loss/(len(features1)*len(features2))

def memory_dissimilar_cosine (bank, features2):
    Loss = 0
    for i in range(len(bank)):
        for j in range(len(features2)):            
            Loss += torch.mean(torch.abs(torch.cosine_similarity(bank[i], features2[j].to(bank[i].device), dim=0)))
    return Loss/(len(bank)*len(features2))

def contrastive_loss(anchor, positive, negatives, temperature=0.07):
    # ensure 2D
    if anchor.dim() == 1:
        anchor = anchor.unsqueeze(0)
    if positive.dim() == 1:
        positive = positive.unsqueeze(0)

    # normalize
    anchor = F.normalize(anchor, dim=1).view(anchor.size(0), -1)
    positive = F.normalize(positive, dim=1).view(positive.size(0), -1)
    negatives = F.normalize(negatives, dim=1).view(negatives.size(0), -1)
    # shape check
    assert anchor.size(1) == negatives.size(1), \
        f"Feature mismatch: {anchor.size(1)} vs {negatives.size(1)}"

    pos_sim = torch.sum(anchor * positive, dim=1) / temperature
    neg_sim = anchor @ negatives.T / temperature

    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)

    return F.cross_entropy(logits, labels)

