# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def unnormalize_image(image):
    # Assuming 'normalized_image' is your normalized image
    mean = (0.48145466, 0.4578275, 0.40821073)
    std_dev = (0.26862954, 0.26130258, 0.27577711)
    unnormalized_image = image * std_dev + mean
    return unnormalized_image


def fit_shapes_to_box(box, shape, withangle=True):
    box = box.detach().cpu().numpy()
    shape = shape.detach().cpu().numpy()
    # mean_before = shape.mean(axis=0)
    shape = remove_outliers(shape)
    # mean_after = shape.mean(axis=0)
    if withangle:
        w, l, h, cx, cy, cz, z = box
    else:
        w, l, h, cx, cy, cz = box
    # scale
    shape_size = np.max(shape, axis=0) - np.min(shape, axis=0)

    shape = shape / shape_size
    shape *= box[:3]
    if withangle:
        # rotate
        shape = (get_rotation(z, degree=False).astype("float32") @ shape.T).T
    # translate
    shape += [cx, cy, cz]
    # shape += (mean_before-mean_after)

    return shape


def get_rotation(z, degree=False):
    if degree:
        z = np.deg2rad(z)
    rot = np.array([[np.cos(z), -np.sin(z),  0],
                    [np.sin(z),  np.cos(z),  0],
                    [0,          0,  1]])
    return rot


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def read_class(path):
    file = open(os.path.join(CONF.PATH.DATA, path), 'r')
    category = file.readline()[:-1]
    word_dict = []
    while category:
        word_dict.append(category)
        category = file.readline()[:-1]

    return word_dict


def params_to_8points_torch(box):
    w, l, h, cx, cy, cz, z = box[:, 0], box[:, 1], box[:, 2], box[:, 3], box[:, 4], box[:, 5], box[:, 6]
    x_corners = torch.stack([w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2], dim=1)
    y_corners = torch.stack([l/2, l/2, l/2, l/2, -l/2, -l/2, -l/2, -l/2], dim=1)
    z_corners = torch.stack([h/2, -h/2, -h/2, h/2, h/2, -h/2, -h/2, h/2], dim=1)
    corners = torch.stack([x_corners, y_corners, z_corners], dim=-1)

    z_rad = z  # ((z/360)*(2*np.pi))

    rot = torch.stack([
        torch.stack([torch.cos(z_rad), -torch.sin(z_rad), torch.zeros_like(z_rad)]).T,
        torch.stack([torch.sin(z_rad),  torch.cos(z_rad), torch.zeros_like(z_rad)]).T,
        torch.stack([torch.zeros_like(z_rad), torch.zeros_like(z_rad), torch.ones_like(z_rad)]).T
    ], dim=-1)

    corners = torch.matmul(corners, rot)
    corners = torch.stack([corners[..., 0]+cx.unsqueeze(1), corners[..., 1]+cy.unsqueeze(1), corners[..., 2]+cz.unsqueeze(1)], dim=-1)

    return corners


def vertify(Z):
    verts = [[Z[0], Z[1], Z[2], Z[3]],
             [Z[4], Z[5], Z[6], Z[7]],
             [Z[0], Z[1], Z[5], Z[4]],
             [Z[2], Z[3], Z[7], Z[6]],
             [Z[1], Z[2], Z[6], Z[5]],
             [Z[4], Z[7], Z[3], Z[0]]]
    return verts


def median(x):
    m, n = x.shape
    middle = np.arange((m-1) >> 1, (m >> 1)+1)
    x = np.partition(x, middle, axis=0)
    return x[middle].mean(axis=0)

# main function


def remove_outliers(data, thresh=3.0):
    m = median(data)
    s = np.abs(data-m)
    new_data = data[(s < median(s)*thresh).all(axis=1)]
    return new_data if new_data.shape[0] > 0 else data


def inverse_homogeneoux_matrix(M):
    R = M[0:3, 0:3]
    T = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -(R.T).dot(T)

    return M_inv


def transform_to_matplotlib_frame(cMo, X, inverse=False):
    M = np.identity(4)
    M[1, 1] = 0
    M[1, 2] = 1
    M[2, 1] = -1
    M[2, 2] = 0

    if inverse:
        return M.dot(inverse_homogeneoux_matrix(cMo).dot(X))
    else:
        return M.dot(cMo.dot(X))



def plot_3Dscene(shapes, bboxes, corners):
    colors = list(mcolors.TABLEAU_COLORS.keys())
    colors_hex = list(mcolors.TABLEAU_COLORS.values())
    pcls = []
    colors_points = []
    corner_points = []
    for i, shape in enumerate(shapes):

        corners[i] = params_to_8points_torch(bboxes[i][None])
        corners_ = corners[i]

        corner_points.append(corners_)
        verts = vertify(corners_)

        pcl = fit_shapes_to_box(bboxes[i][:6], shape[:, :3], withangle=False)

        color = matplotlib.colors.to_rgb(colors_hex[i])
        # ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors="black", alpha=.00))
        pcls.append(pcl)
        colors_points.append(np.tile(color, (pcl.shape[0], 1)))
    pcls = np.concatenate(pcls, axis=0)
    colors_points = np.concatenate(colors_points, axis=0)
    corner_points = np.concatenate(corner_points, axis=0)

    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')

    ax.scatter(pcls[:, 0], pcls[:, 1], pcls[:, 2], c=colors_points, marker=".")

    set_axes_equal(ax)
    plt.show()


def plot_3Dscene_imgs(shapes, bboxes, corners, imgs):
    colors = list(mcolors.TABLEAU_COLORS.keys())
    colors_hex = list(mcolors.TABLEAU_COLORS.values())
    pcls = []
    colors_points = []
    corner_points = []
    for i, shape in enumerate(shapes):

        corners[i] = params_to_8points_torch(bboxes[i][None])
        corners_ = corners[i]

        corner_points.append(corners_)
        verts = vertify(corners_)

        pcl = fit_shapes_to_box(bboxes[i][:6], shape[:, :3], withangle=False)

        color = matplotlib.colors.to_rgb(colors_hex[i])
        # color = mcolors.to_rgba(mcolors.CSS4_COLORS[colors[i]]
        # ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors="black", alpha=.00))
        pcls.append(pcl)
        colors_points.append(np.tile(color, (pcl.shape[0], 1)))
    pcls = np.concatenate(pcls, axis=0)
    colors_points = np.concatenate(colors_points, axis=0)
    corner_points = np.concatenate(corner_points, axis=0)

    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')

    ax.scatter(pcls[:, 0], pcls[:, 1], pcls[:, 2], c=colors_points, marker=".")

    set_axes_equal(ax)
    f = plt.figure()
    for i in range(len(imgs)):

        ax = f.add_subplot(int(f"33{i+1}"))

        ax.imshow(unnormalize_image(imgs[i].permute(1, 2, 0).numpy()))

    # plt.show()


def plot_image9x9(imgs):
    f = plt.figure()
    axes = f.subplots(9, 9).ravel()
    for i in range(len(imgs)):
        ax = axes[i]
        ax.imshow(unnormalize_image(imgs[i].permute(1, 2, 0).numpy()))

    # plt.show()
