import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

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
    image = np.zeros((nbpixels_x, nbpixels_y)) - 5

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
    res = np.zeros((nbpixels_x, nbpixels_y, cs.shape[-1])) - 5
    for i in range(res.shape[-1]):
        cf = plt.contourf(scale_x, scale_y, cs[:, :, i], levels=levels)
        res[:, :, i] = np.flip(
                contourf_to_array(cf, nbpixels_x, nbpixels_y, scale_x, scale_y), axis=0
            )
        res[:, :, i][np.where(res[:, :, i] < -4)] = np.nan
    plt.close("all")
    return res

def create_bounds():
    bounds_k = {
        'x': {
            'min': 137089.2373932857299224,
            'max': 168249.9520578108495101, 
        },
        'y': {
            'min': 589482.3877100050449371,
            'max': 610702.8749795859912410, 
        },
    }
    bounds_g = {
        'x': {
            'min': 129971.5049754020292312,
            'max': 170784.9834783510013949, 
        },
        'y': {
            'min': 584191.5390384565107524,
            'max': 611985.5710535547696054, 
        },
    }
    for key, value in bounds_k.items():
        bounds_k[key]['delta'] = ((bounds_k[key]['max'] - bounds_k[key]['min']) / (8.94 / 10.22) - (bounds_k[key]['max'] - bounds_k[key]['min'])) / 2
        bounds_g[key]['delta'] = ((bounds_g[key]['max'] - bounds_g[key]['min']) / (8.94 / 10.22) - (bounds_g[key]['max'] - bounds_g[key]['min'])) / 2

        bounds_k[key]['min'] -= bounds_k[key]['delta']
        bounds_k[key]['max'] += bounds_k[key]['delta']
        bounds_g[key]['min'] -= bounds_g[key]['delta']
        bounds_g[key]['max'] += bounds_g[key]['delta']
    return bounds_k, bounds_g

def get_bathymetry_extent(ds, bounds):
    print(np.where(ds.x.values >= bounds['x']['min'])[0].min())
    x_index_min = np.where(ds.x.values >= bounds['x']['min'])[0].min() - 1
    x_index_max = np.where(ds.x.values >= bounds['x']['max'])[0].min()
    y_index_min = np.where(ds.y.values >= bounds['y']['min'])[0].max() + 1
    y_index_max = np.where(ds.y.values >= bounds['y']['max'])[0].max()

    extent = (x_index_min, x_index_max, y_index_min, y_index_max)
    return extent

def prep_bathymetry_data(ds, extent):
    x_b, y_b = np.array(ds.x[extent[0]:extent[1]]), np.array(ds.y[extent[3]:extent[2]])
    bodem = np.array(ds[0, extent[3]:extent[2], extent[0]:extent[1]])
    bodem[np.where(bodem == -9999)] = -43.8
    return x_b, y_b, bodem

def get_conc_extent(ds, x_b, y_b):
    x_min = np.where(x_b >= ds.x.values.min())[0].min() - 1
    x_max = np.where(x_b <= ds.x.values.max())[0].max() + 1
    y_min = np.where(y_b >= ds.y.values.min())[0].max() + 1
    y_max = np.where(y_b <= ds.y.values.max())[0].min() - 1

    extent = (x_min, x_max, y_min, y_max)
    return extent

def rescale_and_fit_ds(ds, ds_bounds, rescale_size1, rescale_size2, axis=0):
    """
    Rescales dataset ds to fit over the satellite image with size rescale_size2
    It also fits the dataset values such that the x and y bounds of the dataset are 
    placed on the right positions over the satellite image
    
    Input: 
        ds - dataset to rescale and fit
        ds_bounds - indices of the bounds of the bathymetry, corresponding to the ds bounds
        rescale_size1 - shape of the bathymetry
        rescale_size2 - shape of the satellite image
        axis - axis of ds over which we want to rescale and fit
    
    Output:
        ds_rescaled - dataset with known values on the right positions over the satellite and 
                      nan's everywhere else
    """
    xmin, ymin = ds_bounds[0]
    xmax, ymax = ds_bounds[1]
    ds_sub = np.zeros(rescale_size1)
    ds_sub[:] = np.nan
    for i in range(ds.shape[axis]):
        ds_inter = cv2.resize(ds[i, :, :], dsize=(xmax - xmin, ymin - ymax), interpolation=cv2.INTER_CUBIC)
        ds_sub[ymax:ymin, xmin:xmax] = ds_inter
        ds_sub2 = np.expand_dims(cv2.resize(ds_sub, dsize=(rescale_size2[1], rescale_size2[0]), interpolation=cv2.INTER_CUBIC), axis=axis)
        if i == 0:
            ds_rescaled = ds_sub2
        else:
            ds_rescaled = np.concatenate([ds_rescaled, ds_sub2], axis=axis)
    return ds_rescaled

def create_shaded_image(sat, bodem, shape):
    ls = LightSource(azdeg=315, altdeg=45)
    # Create shade using lightsource
    rgb = ls.hillshade(bodem, vert_exag=5, dx=20, dy=20)

    # Scale satellite image to bathymetry shapes
    sat_scaled = cv2.resize(sat, dsize=(bodem.shape[1], bodem.shape[0]), interpolation=cv2.INTER_CUBIC).astype('float64')

    # Add shade to scaled image
    img_shade = ls.shade_rgb(sat_scaled, bodem, vert_exag=5, blend_mode='soft')
    img_shade = cv2.resize(img_shade, dsize=shape, interpolation=cv2.INTER_CUBIC).astype('float64')

    return img_shade


