import os

import SimpleITK as sikt
import numpy as np
import scipy.ndimage
import torch
from skimage.transform import resize
from torch.utils.data import Dataset

from neighboorCode import neighbour_code_to_normals
import  torchio as tio

class CustomImageDataset(Dataset):
    def __init__(self, CTImagePath, labelPath, imgTransform=None, labelTransform=None, TransformWillChangeValue=None):
        self.CTImagePath = CTImagePath
        self.labelPath = labelPath
        self.imgTransform = imgTransform
        self.labelTransform = labelTransform
        self.TransformWillChangeValue = TransformWillChangeValue

    def __len__(self):
        return len(self.CTImagePath)

    def __getitem__(self, idx):
        image = read_image(self.CTImagePath[idx]).astype(np.float32)
        label = read_label(self.labelPath[idx]).astype(np.int16)
        # image =image/np.max(np.abs(image))
        if self.imgTransform:
            image = self.imgTransform(np.expand_dims(image, 0))
            if self.TransformWillChangeValue:
                image = self.TransformWillChangeValue(image)

        if self.labelTransform:
            label = self.labelTransform(np.expand_dims(label, 0))

        image = torch.tensor(image)
        label = torch.tensor(label).squeeze(0)
        return image, label

class CustomUnlabelledDataset(Dataset):
    def __init__(self, CTImagePath, imgTransform=None, TransformWillChangeValue=None):
        self.CTImagePath = CTImagePath
        self.imgTransform = imgTransform
        self.TransformWillChangeValue = TransformWillChangeValue
    def __len__(self):
        return len(self.CTImagePath)

    def __getitem__(self, idx):
        # print(self.CTImagePath[idx])
        image = read_image(self.CTImagePath[idx]).astype(np.float32)
        # image =image/np.max(np.abs(image))
        # if self.imgTransform:
        #     image = self.imgTransform(np.expand_dims(image, 0))
        #     if self.TransformWillChangeValue:
        #         image = self.TransformWillChangeValue(image)
        img1 = torch.tensor(self.imgTransform(image[None]))
        img2 = torch.tensor(self.imgTransform(image[None]))
        # print(img1.shape)
        return [img1, img2]

class CustomValidImageDataset(Dataset):
    def __init__(self, CTImagePath, labelPath, imgTransform=None, labelTransform=None):
        self.CTImagePath = CTImagePath
        self.labelPath = labelPath
        self.imgTransform = imgTransform
        self.labelTransform = labelTransform
 
    def __len__(self):
        return len(self.CTImagePath)

    def __getitem__(self, idx):
        image = read_image(self.CTImagePath[idx]).astype(np.float32)
        if self.labelPath:  # 如果有labelpath 说明不是test dataset 而是valid
            label = read_label(self.labelPath[idx]).astype(np.int16)
            if self.labelTransform:
                label = self.labelTransform(np.expand_dims(label, 0))
        # image =image/np.max(np.abs(image))
        if self.imgTransform:
            image = self.imgTransform(np.expand_dims(image, 0))

        image = torch.tensor(image)
        if not self.labelPath:  # 如果没有labelpath 直接返回image就行了
            return image
        label = torch.tensor(label).squeeze(0)
        return image, label


def read_image(CTImagePath):
    img = sikt.ReadImage(CTImagePath)
    img = sikt.GetArrayFromImage(img)
    return img


def read_label(labelPath):
    label = sikt.ReadImage(labelPath)
    label = sikt.GetArrayFromImage(label)
    return label


def resizeFun(img, targetSize=(128, 128, 128)):
    return resize(img, output_shape=targetSize, order=0, preserve_range=True)


def compute_surface_distances(mask_gt, mask_pred, spacing_mm):
    """Compute closest distances from all surface points to the other surface.

    Finds all surface elements "surfels" in the ground truth mask `mask_gt` and
    the predicted mask `mask_pred`, computes their area in mm^2 and the distance
    to the closest point on the other surface. It returns two sorted lists of
    distances together with the corresponding surfel areas. If one of the masks
    is empty, the corresponding lists are empty and all distances in the other
    list are `inf` 

    Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.
    spacing_mm: 3-element list-like structure. Voxel spacing in x0, x1 and x2
        direction 

    Returns:
    A dict with 
    "distances_gt_to_pred": 1-dim numpy array of type float. The distances in mm
        from all ground truth surface elements to the predicted surface, 
        sorted from smallest to largest
    "distances_pred_to_gt": 1-dim numpy array of type float. The distances in mm
        from all predicted surface elements to the ground truth surface, 
        sorted from smallest to largest 
    "surfel_areas_gt": 1-dim numpy array of type float. The area in mm^2 of 
        the ground truth surface elements in the same order as 
        distances_gt_to_pred
    "surfel_areas_pred": 1-dim numpy array of type float. The area in mm^2 of 
        the predicted surface elements in the same order as 
        distances_pred_to_gt

    """

    # compute the area for all 256 possible surface elements
    # (given a 2x2x2 neighbourhood) according to the spacing_mm
    neighbour_code_to_surface_area = np.zeros([256])
    for code in range(256):
        normals = np.array(neighbour_code_to_normals[code])
        sum_area = 0
        for normal_idx in range(normals.shape[0]):
            # normal vector
            n = np.zeros([3])
            n[0] = normals[normal_idx, 0] * spacing_mm[1] * spacing_mm[2]
            n[1] = normals[normal_idx, 1] * spacing_mm[0] * spacing_mm[2]
            n[2] = normals[normal_idx, 2] * spacing_mm[0] * spacing_mm[1]
            area = np.linalg.norm(n)
            sum_area += area
    neighbour_code_to_surface_area[code] = sum_area

    # compute the bounding box of the masks to trim
    # the volume to the smallest possible processing subvolume
    mask_all = mask_gt | mask_pred
    bbox_min = np.zeros(3, np.int64)
    bbox_max = np.zeros(3, np.int64)

    # max projection to the x0-axis
    proj_0 = np.max(np.max(mask_all, axis=2), axis=1)
    idx_nonzero_0 = np.nonzero(proj_0)[0]
    if len(idx_nonzero_0) == 0:
        return {"distances_gt_to_pred": np.array([]),
                "distances_pred_to_gt": np.array([]),
                "surfel_areas_gt": np.array([]),
                "surfel_areas_pred": np.array([])}

    bbox_min[0] = np.min(idx_nonzero_0)
    bbox_max[0] = np.max(idx_nonzero_0)

    # max projection to the x1-axis
    proj_1 = np.max(np.max(mask_all, axis=2), axis=0)
    idx_nonzero_1 = np.nonzero(proj_1)[0]
    bbox_min[1] = np.min(idx_nonzero_1)
    bbox_max[1] = np.max(idx_nonzero_1)

    # max projection to the x2-axis
    proj_2 = np.max(np.max(mask_all, axis=1), axis=0)
    idx_nonzero_2 = np.nonzero(proj_2)[0]
    bbox_min[2] = np.min(idx_nonzero_2)
    bbox_max[2] = np.max(idx_nonzero_2)

    print("bounding box min = {}".format(bbox_min))
    print("bounding box max = {}".format(bbox_max))

    # crop the processing subvolume.
    # we need to zeropad the cropped region with 1 voxel at the lower, 
    # the right and the back side. This is required to obtain the "full" 
    # convolution result with the 2x2x2 kernel
    cropmask_gt = np.zeros((bbox_max - bbox_min) + 2, np.uint8)
    cropmask_pred = np.zeros((bbox_max - bbox_min) + 2, np.uint8)

    cropmask_gt[0:-1, 0:-1, 0:-1] = mask_gt[bbox_min[0]:bbox_max[0] + 1,
                                    bbox_min[1]:bbox_max[1] + 1,
                                    bbox_min[2]:bbox_max[2] + 1]

    cropmask_pred[0:-1, 0:-1, 0:-1] = mask_pred[bbox_min[0]:bbox_max[0] + 1,
                                      bbox_min[1]:bbox_max[1] + 1,
                                      bbox_min[2]:bbox_max[2] + 1]

    # compute the neighbour code (local binary pattern) for each voxel
    # the resultsing arrays are spacially shifted by minus half a voxel in each axis.
    # i.e. the points are located at the corners of the original voxels
    kernel = np.array([[[128, 64],
                        [32, 16]],
                       [[8, 4],
                        [2, 1]]])
    neighbour_code_map_gt = scipy.ndimage.filters.correlate(cropmask_gt.astype(np.uint8), kernel, mode="constant",
                                                            cval=0)
    neighbour_code_map_pred = scipy.ndimage.filters.correlate(cropmask_pred.astype(np.uint8), kernel, mode="constant",
                                                              cval=0)

    # create masks with the surface voxels
    borders_gt = ((neighbour_code_map_gt != 0) & (neighbour_code_map_gt != 255))
    borders_pred = ((neighbour_code_map_pred != 0) & (neighbour_code_map_pred != 255))

    # compute the distance transform (closest distance of each voxel to the surface voxels)
    if borders_gt.any():
        distmap_gt = scipy.ndimage.morphology.distance_transform_edt(~borders_gt, sampling=spacing_mm)
    else:
        distmap_gt = np.Inf * np.ones(borders_gt.shape)

    if borders_pred.any():
        distmap_pred = scipy.ndimage.morphology.distance_transform_edt(~borders_pred, sampling=spacing_mm)
    else:
        distmap_pred = np.Inf * np.ones(borders_pred.shape)

    # compute the area of each surface element
    surface_area_map_gt = neighbour_code_to_surface_area[neighbour_code_map_gt]
    surface_area_map_pred = neighbour_code_to_surface_area[neighbour_code_map_pred]

    # create a list of all surface elements with distance and area
    distances_gt_to_pred = distmap_pred[borders_gt]
    distances_pred_to_gt = distmap_gt[borders_pred]
    surfel_areas_gt = surface_area_map_gt[borders_gt]
    surfel_areas_pred = surface_area_map_pred[borders_pred]

    # sort them by distance
    if distances_gt_to_pred.shape != (0,):
        sorted_surfels_gt = np.array(sorted(zip(distances_gt_to_pred, surfel_areas_gt)))
        distances_gt_to_pred = sorted_surfels_gt[:, 0]
        surfel_areas_gt = sorted_surfels_gt[:, 1]

    if distances_pred_to_gt.shape != (0,):
        sorted_surfels_pred = np.array(sorted(zip(distances_pred_to_gt, surfel_areas_pred)))
        distances_pred_to_gt = sorted_surfels_pred[:, 0]
        surfel_areas_pred = sorted_surfels_pred[:, 1]

    return {"distances_gt_to_pred": distances_gt_to_pred,
            "distances_pred_to_gt": distances_pred_to_gt,
            "surfel_areas_gt": surfel_areas_gt,
            "surfel_areas_pred": surfel_areas_pred}


def compute_average_surface_distance(surface_distances):
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    average_distance_gt_to_pred = np.sum(distances_gt_to_pred * surfel_areas_gt) / np.sum(surfel_areas_gt)
    average_distance_pred_to_gt = np.sum(distances_pred_to_gt * surfel_areas_pred) / np.sum(surfel_areas_pred)
    return (average_distance_gt_to_pred, average_distance_pred_to_gt)


def compute_robust_hausdorff(surface_distances, percent):
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    if len(distances_gt_to_pred) > 0:
        surfel_areas_cum_gt = np.cumsum(surfel_areas_gt) / np.sum(surfel_areas_gt)
        idx = np.searchsorted(surfel_areas_cum_gt, percent / 100.0)
        perc_distance_gt_to_pred = distances_gt_to_pred[min(idx, len(distances_gt_to_pred) - 1)]
    else:
        perc_distance_gt_to_pred = np.Inf

    if len(distances_pred_to_gt) > 0:
        surfel_areas_cum_pred = np.cumsum(surfel_areas_pred) / np.sum(surfel_areas_pred)
        idx = np.searchsorted(surfel_areas_cum_pred, percent / 100.0)
        perc_distance_pred_to_gt = distances_pred_to_gt[min(idx, len(distances_pred_to_gt) - 1)]
    else:
        perc_distance_pred_to_gt = np.Inf

    return max(perc_distance_gt_to_pred, perc_distance_pred_to_gt)


def compute_surface_overlap_at_tolerance(surface_distances, tolerance_mm):
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    rel_overlap_gt = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm]) / np.sum(surfel_areas_gt)
    rel_overlap_pred = np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm]) / np.sum(surfel_areas_pred)
    return (rel_overlap_gt, rel_overlap_pred)


def compute_surface_dice_at_tolerance(surface_distances, tolerance_mm):
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    overlap_gt = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm])
    overlap_pred = np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm])
    surface_dice = (overlap_gt + overlap_pred) / (
            np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred))
    return surface_dice


def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.

    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`. 

    Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


if __name__ == '__main__':
    # mask_gt   = np.ones((5,1,128,128,128), np.uint8)
    # mask_pred = np.ones((5,1,128,128,128), np.uint8)
    # mask_gt[0,0,50,60,70] = 2
    # # mask_pred[50,60,72] = 1
    # # surface_distances = compute_surface_distances(mask_gt, mask_pred, spacing_mm=(3,2,1))
    # # print("surface dice at 1mm:      {}".format(compute_surface_dice_at_tolerance(surface_distances, 1)))
    # print("volumetric dice:          {}".format(compute_dice_coefficient(mask_gt, mask_pred)))

    dataDirPath = "data/FLARE22_LabeledCase50-20220324T003930Z-001"
    imgPaths = list(
        map(lambda x: os.path.join(dataDirPath, "images", x), os.listdir(os.path.join(dataDirPath, "images"))))
    labelPath = list(
        map(lambda x: os.path.join(dataDirPath, "labels", x), os.listdir(os.path.join(dataDirPath, "labels"))))
    print(imgPaths)
    print(labelPath)
    splitIndex = int(len(imgPaths) * 0.8)
    trainDataset = CustomImageDataset(CTImagePath=imgPaths[0:splitIndex],
                                      labelPath=labelPath[0:splitIndex],
                                      labelTransform=resizeFun,
                                      imgTransform=resizeFun)
    for img, label in trainDataset:
        print(img.size(), label.size())
        break
    print("total images:", len(trainDataset))
