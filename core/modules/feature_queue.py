import torch


class FeatureQueue:
    def __init__(self, config, size):
        self.bz = config['batchsize']
        self.label_bs = config['label_bs']
        self.size = size
        self.feature_list = []
        self.label_list = []
        self.label_mask_list = [j+self.bz*i for i in range(self.size) for j in range(self.label_bs)]

    def enqueue(self, tensor, label):
        if len(self.feature_list) >= self.size:
            self.dequeue()
        if len(self.feature_list)>1:
            self.feature_list[-1].detach()
        self.feature_list.append(tensor)
        self.label_list.append(label)

    def dequeue(self):
        self.feature_list.pop(0)
        self.label_list.pop(0)

    def get_feature(self):
        return torch.cat(self.feature_list, dim=0)
    
    def get_label(self):
        current_size = len(self.feature_list)
        return torch.cat(self.label_list, dim=0),self.label_mask_list[:self.label_bs*current_size]

