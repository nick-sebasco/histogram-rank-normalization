# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 07:30:11 2022

@author: mawes
@editor: sebasco

General notes:
Fix output directory.
"""
import zarr
import numpy as np
import os
import scipy.stats
import cv2
from sys import argv
from typing import List


def RankNormalize(score_vector, dist, loc, scale):
    sort_idx = score_vector.argsort()
    rank_vector = sort_idx.argsort()
    rank_vector = (rank_vector + 0.5) / len(rank_vector)
    norm_score_vector = dist.ppf(rank_vector, loc=loc, scale=scale)
    return norm_score_vector


def OtsuMethod(score_vector, num_bins):
    score_max = np.max(score_vector)
    score_min = np.min(score_vector)
    bins = np.linspace(score_min, score_max, num_bins)
    thresh = score_min
    max_val = 0
    for tt in range(1, num_bins - 1):
        below_thresh = score_vector < bins[tt]
        above_thresh = score_vector >= bins[tt]
        wB = np.sum(below_thresh).astype(np.double)
        wF = np.sum(above_thresh).astype(np.double)
        mB = np.mean(score_vector[below_thresh]).astype(np.double)
        mF = np.mean(score_vector[above_thresh]).astype(np.double)
        val = wF * wB * (mB - mF) ** 2
        if val > max_val:
            thresh = bins[tt]
    return thresh


def combined_age_score(
    output_score_path: str,
    combined_score_path: str,
    image_dir=[],
    thresh_method="GaussianBlur",
    thresh_param=200,
    dist=scipy.stats.norm,
    loc=0,
    scale=1,
    **_,
) -> List:
    """
    TODO: This algorithm is kinda slow.
    :param output_score_path: This is the path to the directory of the zarr scores.  The format of the path must be
        '\s/level 1/...(/)\s'.  With the last / being optional and an arbitrary number of leading and trailing spaces.
    :param combined_score_path: This is the path to where the combined scores will be saved.
    :param image_dir: This is the path to the source images.

    :return: A 2-component list consisting of the bounds of the age scores.
    """
    output_score_path = output_score_path.strip()
    output_score_path += "/" if output_score_path[-1] != "/" else ""
    score_mask_zarrs = os.listdir(output_score_path)

    score_mask_zarrs = [
        score_mask for score_mask in score_mask_zarrs if ".zarr" in score_mask
    ]
    score_mask_zarrs = np.sort(score_mask_zarrs)
    zm = zarr.open(os.path.join(output_score_path, score_mask_zarrs[0]))

    if thresh_method == "GaussianBlur":
        images = os.listdir(image_dir)
        print("i. images:", images)
        images = np.sort(images)
        im_path = os.path.join(image_dir, images[0])
        print("i. im_path:", im_path)
        image = cv2.imread(im_path)

        if image.shape[0] < zm.shape[0]:
            image = np.vstack(
                (
                    image,
                    np.ones((zm.shape[0] - image.shape[0], image.shape[1], 3), np.uint8)
                    * 255,
                )
            )
        if image.shape[1] < zm.shape[1]:
            image = np.hstack(
                (
                    image,
                    np.ones((image.shape[0], zm.shape[1] - image.shape[1], 3), np.uint8)
                    * 255,
                )
            )
        image = cv2.GaussianBlur(image, (9, 9), 0)
        green_channel = image[:, :, 1].flatten()
        print("GREEN CHANNEL LENGTH:", len(green_channel),  zm.shape)

    ch_0_scores = zm[:, :, 0, :]
    ch_1_scores = zm[:, :, 1, :]
    ch_2_scores = zm[:, :, 2, :]
    score_shape = [0, ch_0_scores.shape]
    ch_0_scores = ch_0_scores.flatten()
    ch_1_scores = ch_1_scores.flatten()
    ch_2_scores = ch_2_scores.flatten()
    score_idx = [0, len(ch_0_scores)]

    # num_patches=[zm.shape[3]]

    for ii in range(1, len(score_mask_zarrs)):
        file = score_mask_zarrs[ii]

        # Error generated here again.  This function should enforce path format.
        # Fixing output_score_path addresses this.
        print(f"processing {ii}/{len(score_mask_zarrs)}")
        zm = zarr.open(os.path.join(output_score_path, file))

        ch_0_scores_sub = zm[:, :, 0, :]
        ch_1_scores_sub = zm[:, :, 1, :]
        ch_2_scores_sub = zm[:, :, 2, :]
        score_shape.append(ch_0_scores_sub.shape)
        ch_0_scores_sub = ch_0_scores_sub.flatten()
        ch_1_scores_sub = ch_1_scores_sub.flatten()
        ch_2_scores_sub = ch_2_scores_sub.flatten()
        score_idx.append(score_idx[-1] + len(ch_0_scores_sub))
        ch_0_scores = np.concatenate((ch_0_scores, ch_0_scores_sub), axis=0)
        ch_1_scores = np.concatenate((ch_1_scores, ch_1_scores_sub), axis=0)
        ch_2_scores = np.concatenate((ch_2_scores, ch_2_scores_sub), axis=0)
        if thresh_method == "GaussianBlur":
            im_path = os.path.join(image_dir, images[ii])
            image = cv2.imread(im_path)

            if image.shape[0] < zm.shape[0]:
                image = np.vstack(
                    (
                        image,
                        np.ones(
                            (zm.shape[0] - image.shape[0], image.shape[1], 3), np.uint8
                        )
                        * 255,
                    )
                )
            if image.shape[1] < zm.shape[1]:
                image = np.hstack(
                    (
                        image,
                        np.ones(
                            (image.shape[0], zm.shape[1] - image.shape[1], 3), np.uint8
                        )
                        * 255,
                    )
                )
            image = cv2.GaussianBlur(image, (9, 9), 0)
            green_channel = np.concatenate(
                (green_channel, image[:, :, 1].flatten()), axis=0
            )
            print("GREEN CHANNEL CONCAT LENGTH:", len(green_channel), zm.shape)
        # num_patches.append(zm.shape[3])

    if thresh_method == "GaussianBlur":
        foreground = green_channel <= thresh_param
    elif thresh_method == "Otsu":
        thresh = OtsuMethod(ch_0_scores, thresh_param)
        foreground = ch_0_scores >= thresh
    elif thresh_method == "Percentile":
        thresh = np.percentile(ch_0_scores, thresh_param)
        foreground = ch_0_scores >= thresh
    else:
        thresh = thresh_method(ch_0_scores, thresh_param)
        foreground = ch_0_scores >= thresh

    adj_ch_0_scores = RankNormalize(ch_0_scores[foreground], dist, loc, scale)
    adj_ch_1_scores = RankNormalize(ch_1_scores[foreground], dist, loc, scale)
    adj_ch_2_scores = RankNormalize(ch_2_scores[foreground], dist, loc, scale)
    ch_0_scores[~foreground] = (
        0  # np.min(adj_ch_0_scores)-(np.max(adj_ch_0_scores)-np.min(adj_ch_0_scores))/1000
    )
    ch_1_scores[~foreground] = (
        0  # np.min(adj_ch_1_scores)-(np.max(adj_ch_1_scores)-np.min(adj_ch_1_scores))/1000
    )
    ch_2_scores[~foreground] = (
        0  # np.min(adj_ch_2_scores)-(np.max(adj_ch_2_scores)-np.min(adj_ch_2_scores))/1000
    )
    ch_0_scores[foreground] = adj_ch_0_scores
    ch_1_scores[foreground] = adj_ch_1_scores
    ch_2_scores[foreground] = adj_ch_2_scores
    flat_age_scores = (ch_0_scores + (ch_1_scores) + (ch_2_scores)) / 3

    for ff in range(1, len(score_idx)):
        sub_age_scores = flat_age_scores[score_idx[ff - 1] : score_idx[ff]]
        sub_age_scores = np.reshape(sub_age_scores, score_shape[ff])

        # TODO: unsafe path concatenation.
        zarr.save(
            combined_score_path
            + score_mask_zarrs[ff - 1].replace(".zarr", "_age_scores.zarr"),
            sub_age_scores,
        )
    array_bounds = [
        np.min(flat_age_scores[foreground]),
        np.max(flat_age_scores[foreground]),
    ]

    return array_bounds


if __name__ == "__main__":
    _, image_dir, output_score_path, combined_score_path = argv
    combined_age_score(
        output_score_path,
        combined_score_path,
        image_dir=image_dir,
        thresh_method="GaussianBlur",
        thresh_param=200,
        dist=scipy.stats.norm,
        loc=0,
        scale=1,
    )
