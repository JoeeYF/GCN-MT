import torch
from torch.nn import functional as F


def cls_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = torch.softmax(input_logits, dim=1)
    target_softmax = torch.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    bs = input_logits.size()[0]
    return F.mse_loss(input_softmax, target_softmax, reduction='sum') / (num_classes * bs)


def att_mse_loss(mask, cams):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert mask.size() == cams.size() and len(mask.size()) == 4
    mse_loss = F.mse_loss(mask, cams, reduction='none').sum((2, 3))
    norm = (mask.sum((2, 3)) + cams.sum((2, 3))).sum()
    mse_loss = torch.sum(mse_loss) / torch.clamp(norm, min=1e-5)
    return mse_loss


def relation_mse_loss(activations, ema_activations):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    assert activations.size() == ema_activations.size()

    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    similarity = activations.mm(activations.t())
    norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    norm_similarity = similarity / norm

    ema_similarity = ema_activations.mm(ema_activations.t())
    ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (norm_similarity-ema_norm_similarity)**2
    return similarity_mse_loss
