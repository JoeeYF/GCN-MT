import torch
import os
from models.GCNModel.GAT import GAT, MutilHeadGAT
from models.AttentionNet import AttentionNet

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
StudentModel = AttentionNet('se_resnet50', 7, ema=False).cuda()
TeacherModel = AttentionNet('se_resnet50', 7, ema=True).cuda()
GCNModel = MutilHeadGAT(1024, 512, 7).cuda()
StudentModel = torch.nn.DataParallel(StudentModel)
TeacherModel = torch.nn.DataParallel(TeacherModel)
GCNModel = torch.nn.DataParallel(GCNModel)
print('model done')

image = torch.ones((4, 3, 128, 128)).cuda().float()
print('image', image.shape)

output_s, cam_refined_s, feature_s = StudentModel(image)
output_t, cam_refined_t, feature_t = TeacherModel(image)
print('feature_s', feature_s.shape)
print('cam_refined_s', cam_refined_s.shape)
print('output_s', output_s.shape)
# print('feature_t', feature_t.shape)

adj = torch.matmul(feature_s, feature_t.T)
adj_softmax = torch.softmax(adj,dim=1)
adj = (adj_softmax+adj_softmax.T)/2
feature = torch.cat([feature_s, feature_t], dim=1)
output_gcn = GCNModel(feature, adj)
print('adj', adj.shape)
print('feature', feature.shape)
print('out_gcn', output_gcn.shape)

optimizer = torch.optim.Adam( [{'params': TeacherModel.parameters()}, {'params': GCNModel.parameters()}], 0.011, weight_decay=0.000001)
print(len(optimizer.param_groups))