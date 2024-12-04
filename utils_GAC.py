import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn 
import cv2
import math
import torch.nn.functional as F
import nibabel as nib

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred=pred + 0
    gt = gt + 0
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, jaccard, hd95, asd
    elif pred.sum() > 0 and gt.sum()==0:
        return 0, 0, 10, 10
    else:
        return 0, 0, 10, 10


def test_single_image(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        boundary_prediction = np.zeros_like(image)
        for ind in range(image.shape[2]):
            slice = image[ :, :, ind]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                out_dis, boundary = net(input)
                out = ICSTM(out_dis, boundary, 10).squeeze(0).squeeze(0)
                boundary_out = (boundary.squeeze(0).squeeze(0)).cpu().detach().numpy()
                boundary_out = boundary_out
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                    boundary_pred = zoom(boundary_out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                    boundary_pred = boundary_out
                prediction[:, :, ind] = pred
                boundary_prediction[:, :, ind] = boundary_pred
    else:
        x, y = image.shape[0], image.shape[1]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_dis, boundary = net(input)
            out = ICSTM(out_dis, boundary, 10).squeeze(0).squeeze(0)
            boundary_out = (boundary.squeeze(0).squeeze(0)).cpu().detach().numpy()
            boundary_out = 1*(boundary_out>0.5)
            out = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                prediction = out
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        if len(image.shape) == 3:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)),
                     test_save_path + '/' + case + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)),
                     test_save_path + '/' + case + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)),
                     test_save_path + '/' + case + "_gt.nii.gz")
            nib.save(nib.Nifti1Image(boundary_prediction.astype(np.float32), np.eye(4)),
                     test_save_path + '/' + case + "_boundary.nii.gz")
        else:
            cv2.imwrite(test_save_path + '/'+case + 'pred.png', (255*prediction).astype(np.float32))
            cv2.imwrite(test_save_path + '/' + case + 'image.png', (image).astype(np.float32))
            cv2.imwrite(test_save_path + '/' + case + 'boundary.png', (255*boundary_out).astype(np.float32))
    return metric_list



def ICSTM(pred,boundary,lan):
    u = torch.sigmoid(-1500 *  pred)
    with torch.no_grad():
        b = 1 * (boundary > 0.5)
        g = 1 / (1 + (20 * b) ** 2)
        iterNum = 20
        tau = 2
        kernel = np.multiply(cv2.getGaussianKernel(math.floor(6 * tau / 2) * 2 + 1, tau),
                        (cv2.getGaussianKernel(math.floor(6 * tau / 2) * 2 + 1, tau).T))
        kernel = kernel.astype(np.float32)
        kernel = torch.from_numpy(kernel).cuda()
        kernel = (kernel.unsqueeze(0)).unsqueeze(0)
    for k in range(iterNum):
        varphi = pred +  lan * (torch.sqrt(g) * F.conv2d(torch.sqrt(g) * (1 - 2 * u), kernel,padding=(6,6)))
        u = torch.sigmoid(-1500 * varphi)
    return u