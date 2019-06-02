import torch


def loss_function(pred, target, weights):
    error = 0
    for i in range(pred.size(0)):
        pred_compare = pred[i, target[i][0]-2:target[i][0]+2+1]
        loss = torch.mul(pred_compare, weights).sum()
        error = error - loss

    return error / pred.size(0)


def pixel_accuracy(pred, target, pixel=3):
    pred, _ = pred.max(dim=1)
    pred = pred.abs()
    return pred[pred <= pixel].shape[0] / pred.size(0)


def pixel_accuracy2(pred, target, pixel=3):
    _, indices = pred.max(dim=1)
    target = target.type(torch.long)
    pred = (indices-target).abs()
    batch_size = pred.size(0)
    return pred[pred <= pixel].sum() / batch_size
