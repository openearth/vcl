import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt

def rotate_and_crop(arr, ang):
    """Array arr to be rotated by ang degrees and cropped afterwards"""
    arr_rot = scipy.ndimage.rotate(arr, ang, reshape=True, order=0)

    shift_up = np.ceil(np.arcsin(abs(ang) / 360 * 2 * np.pi) * arr.shape[1])
    shift_right = np.ceil(np.arcsin(abs(ang) / 360 * 2 * np.pi) * arr.shape[0])

    arr_crop = arr_rot[
        int(shift_up) : arr_rot.shape[0] - int(shift_up),
        int(shift_right) : arr_rot.shape[1] - int(shift_right),
    ]

    return arr_crop


def contourf_to_array(cs, nbpixels_x, nbpixels_y, scale_x, scale_y):
    """Draws filled contours from contourf or tricontourf cs on output array of size (nbpixels_x, nbpixels_y)"""
    image = np.zeros((nbpixels_x, nbpixels_y)) - 2

    for i, collection in enumerate(cs.collections):
        z = cs.levels[i]  # get contour levels from cs
        for path in collection.get_paths():
            verts = (
                path.to_polygons()
            )  # get vertices of current contour level (is a list of arrays)
            for v in verts:
                # rescale vertices to image size
                v[:, 0] = (
                    (v[:, 0] - np.min(scale_x))
                    / (np.max(scale_x) - np.min(scale_x))
                    * nbpixels_y
                )
                v[:, 1] = (
                    (v[:, 1] - np.min(scale_y))
                    / (np.max(scale_y) - np.min(scale_y))
                    * nbpixels_x
                )
                poly = np.array(
                    [v], dtype=np.int32
                )  # dtype integer is necessary for the next instruction
                cv2.fillPoly(image, poly, z)
    return image

def contourf_to_array_3d(cs, nbpixels_x, nbpixels_y, scale_x, scale_y, levels):
    res = np.zeros((nbpixels_x, nbpixels_y, cs.shape[-1])) - 2
    for i in range(res.shape[-1]):
        cf = plt.contourf(scale_x, scale_y, cs[:, :, i], levels=levels)
        res[:, :, i] = np.flip(
                contourf_to_array(cf, nbpixels_x, nbpixels_y, scale_x, scale_y), axis=0
            )
        res[:, :, i][np.where(res[:, :, i] < -1)] = np.nan
    plt.close("all")
    return res
