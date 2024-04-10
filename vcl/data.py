import shapely
import rasterio
import pandas as pd
import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt
import rioxarray as rxr
from matplotlib.colors import LightSource


def compute_rotation_angle(extent):
    """
    Function for computing the rotation angle of the geometry extent
    The angle is the counterclockwise angle between the geometry and the x-axis
    """
    # Get vertices of geometry (rectangle)
    coords = list(extent.exterior.coords)
    coords = pd.DataFrame(coords)
    coords.columns = ["x", "y"]

    # Get minimum and maximum x and y values from vertices
    xmin = coords.idxmin(0)["x"]
    ymin = coords.idxmin(0)["y"]
    xmax = coords.idxmax(0)["x"]
    ymax = coords.idxmax(0)["y"]

    # Find bottom left point and bottom right point of original extent
    bottom_point = [coords.iloc[ymin]["x"], coords.iloc[ymin]["y"]]
    right_point = [coords.iloc[xmax]["x"], coords.iloc[xmax]["y"]]

    # Compute the rotation angle of original extent
    o = right_point[1] - bottom_point[1]
    a = right_point[0] - bottom_point[0]

    angle = np.rad2deg(np.arctan(o / a))

    return angle


def rotate_and_crop(arr, ang, cval=np.nan):
    """Array arr to be rotated by ang degrees and cropped afterwards"""
    arr_rot = scipy.ndimage.rotate(arr, ang, reshape=True, order=0, cval=cval)

    shift_up = np.ceil(np.arcsin(abs(ang) / 360 * 2 * np.pi) * arr.shape[1])
    shift_right = np.ceil(np.arcsin(abs(ang) / 360 * 2 * np.pi) * arr.shape[0])

    arr_crop = arr_rot[
        int(shift_up) : arr_rot.shape[0] - int(shift_up),
        int(shift_right) : arr_rot.shape[1] - int(shift_right),
    ]

    return arr_rot


def contourf_to_array(cs, nbpixels_x, nbpixels_y, scale_x, scale_y):
    """
    Draws filled contours from contourf or tricontourf cs on output array of size (nbpixels_x, nbpixels_y)

    Input:
        - cs, a contourf or tricontourf object from matplotlib
        - nbpixles_x, the number of rows for the output arrays
        - nbpixels_y, the number of columns for the output arrays
        - scale_x, the x component of the meshgrid for the contours
        - scale_y, the y component of the meshgrid for the contours

    Output:
        - images, a dictionary of arrays for each contour level
        - shapes, a dictionary of polygons from the contour for each contour level
    """
    image = np.zeros((nbpixels_x, nbpixels_y)) - 10
    images = {}
    shapes = {}

    for i, collection in enumerate(cs.collections):
        z = cs.levels[i]  # get contour levels from cs
        images[i] = np.full((nbpixels_x, nbpixels_y), np.nan)
        for path in collection.get_paths():
            verts = (
                path.to_polygons()
            )  # get vertices of current contour level (is a list of arrays)
            for j, v in enumerate(verts):
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
                cv2.fillPoly(images[i], poly, z)
                if j == 0:
                    shapes[i] = shapely.Polygon(poly[0, ...])
                    shapes[i] = shapely.MultiPolygon([shapes[i]])
                else:
                    try:
                        shapes[i] = shapes[i].union(shapely.Polygon(poly[0, ...]))
                    except:
                        continue
    return images, shapes


# This function was needed, since otherwise a smaller polygon fully inside a bigger polygon could get overwritten
def combine_polygons(images, shapes):
    """
    Function for combining images containing polygons from contour levels
    into one image in the order of biggest polygons to smallest polygons

    Input:
        - images, a dictionary of arrays for each contour level
        - shapes, a dictionary of polygons from the contour for each contour level

    Output:
        - image, an array combined from images
    """

    # Sort polygons from big to small
    polygons = sorted(shapes.items(), key=lambda item: item[1].area, reverse=True)
    image = np.full(images[0].shape, np.nan)
    # Add polygons from images to image
    for i in polygons:
        image[~np.isnan(images[i[0]])] = np.nanmax(images[i[0]])

    return image


def contourf_to_array_3d(
    cs, nbpixels_x, nbpixels_y, nbpixels_z, axis, scale_x, scale_y, levels
):
    """
    Draws filled contours for 3d array cs along the specified axis
    on output array of size (nbpixels_x, nbpixels_y, nbpixels_z)

    Input:
        - cs, 3 dimensional array
        - nbpixles_x, the number of rows for the output array
        - nbpixels_y, the number of columns for the output array
        - nbpixels_z, the number of drawn contours of shape (nbpixels_x, nbpixels_y) in the output array
        - axis, the axis along which the filled contours need to be created
        - scale_x, the x component of the meshgrid for the contours
        - scale_y, the y component of the meshgrid for the contours
        - levels, the contour levels

    Output:
        - res, a 3 dimensional array of shape (nbpixels_x, nbpixels_y, nbpixels_z) with nbpixels_z amount of contours
    """

    res = np.zeros((nbpixels_x, nbpixels_y, nbpixels_z)) - 10
    res = np.full((nbpixels_x, nbpixels_y, nbpixels_z), np.nan)
    for i in range(res.shape[-1]):
        # create a full slice
        s = [slice(None)] * res.ndim
        # replace the slice on the desired axis
        s[axis] = i
        indexing = tuple(s)
        cf = plt.contourf(scale_x, scale_y, cs[indexing], levels=levels)
        contourfs, shapes = contourf_to_array(
            cf, nbpixels_x, nbpixels_y, scale_x, scale_y
        )
        res[..., i] = np.flip(combine_polygons(contourfs, shapes), axis=0)
    plt.close("all")
    return res


def sat_and_bodem_bounds(sat, bodem):
    # Get bounds of bodem and sat files
    sat_bounds = sat.bounds
    bodem_bounds = bodem.bounds

    # Compute the minimum (and maximum) point which sat and bodem still share in their extent
    left = max([sat_bounds[0], bodem_bounds[0]])
    bottom = max([sat_bounds[1], bodem_bounds[1]])
    right = min([sat_bounds[2], bodem_bounds[2]])
    top = min([sat_bounds[3], bodem_bounds[3]])

    return left, right, bottom, top


def contour_bounds(ds):
    left = ds.x.values.min()
    right = ds.x.values.max()
    bottom = ds.y.values.min()
    top = ds.y.values.max()

    return left, right, bottom, top


def create_shaded_image(sat_extent, bodem):
    # Get satellite image
    sat = sat_extent.read()
    sat = np.transpose(sat, (1, 2, 0))

    # Get bounds of bodem and sat files
    left, right, bottom, top = sat_and_bodem_bounds(sat_extent, bodem)

    # Get transform of the files (for indices instead of coordinates)
    t_sat = sat_extent.transform
    t_bodem = bodem.transform

    # Apply inverse transform to extrema to get indices of arrays
    window_bodem_min = ~t_bodem * (left, top)
    window_bodem_max = ~t_bodem * (right, bottom)

    window_sat_min = ~t_sat * (left, top)
    window_sat_max = ~t_sat * (right, bottom)

    bodem = bodem.read(1)

    # Get extent of arrays from indices
    bodem = bodem[
        int(window_bodem_min[1]) : int(window_bodem_max[1]),
        int(window_bodem_min[0]) : int(window_bodem_max[0]),
    ]
    sat = sat[
        int(window_sat_min[1]) : int(window_sat_max[1]),
        int(window_sat_min[0]) : int(window_sat_max[0]),
    ]

    # Scale satellite image to bathymetry shapes
    sat_scaled = (
        cv2.resize(
            sat, dsize=(bodem.shape[1], bodem.shape[0]), interpolation=cv2.INTER_CUBIC
        ).astype("float64")
        / 255
    )

    ls = LightSource(azdeg=315, altdeg=45)

    # Add shade to scaled image
    img_shade = ls.shade_rgb(sat_scaled, bodem, vert_exag=5, blend_mode="soft")

    return img_shade


def get_plot_lims(extent):
    lims = extent.exterior.bounds
    lims = list(lims)
    lims[0] += 750
    lims[1] -= 900
    lims[2] += 750
    lims[3] -= 900
    return tuple(lims)


def prepare_rasterio_image(src, transform=None):
    image = src.read()
    image = np.transpose(image, (1, 2, 0))

    bounds = src.bounds

    if transform is not None:
        xmin, ymin = transform.transform(bounds[0], bounds[1])
        xmax, ymax = transform.transform(bounds[2], bounds[3])
        bounds = (xmin, ymin, xmax, ymax)

    return image, bounds


def get_rotated_vertex(center, point, angle):
    """
    Computes new coordinates of a rectangle vertex after rotation

    Input:
        - center, the center point of the rectangle
        - point, the original vertex point
        - angle, the rotation angle of the rectangle in radians

    Output:
        - rotated_point, the new coordinates of point after rotation
    """
    # Translate point such that 'rectangle' has center at (0, 0)
    new_point = np.array(point) - np.array(center)
    # Create rotation matrix
    rot_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    # Compute rotated point
    rotated_point = np.matmul(rot_matrix, new_point)
    # translate point back
    rotated_point = rotated_point + np.array(center)
    return rotated_point


def rotate_1d_array(center, x, y, angle):
    """
    Computes new coordinates of x and y after rotation

    Input:
        - center, the center point of the rectangle
        - x, the original x points
        - y, the original y points
        - angle, the rotation angle of the rectangle in radians

    Output:
        - rotated_x, the new coordinates the x points after rotation
        - rotated_y, the new coordinates the y points after rotation
    """
    x_shift = center[0]
    y_shift = center[1]

    x_rotated = (x - x_shift) * np.cos(angle) - (y - y_shift) * np.sin(angle)
    y_rotated = (x - x_shift) * np.sin(angle) + (y - y_shift) * np.cos(angle)

    x_rotated += x_shift
    y_rotated += y_shift

    return x_rotated, y_rotated


def fit_rot_ds_to_bounds(ds, rot_ds, center, extent, angle):
    left = ds.conc.x.values[0]
    bottom = ds.conc.y.values[-1]
    right = ds.conc.x.values[-1]
    top = ds.conc.y.values[0]

    # Compute new vertex positions after rotation
    xmin1 = get_rotated_vertex(center, (left, bottom), np.deg2rad(angle))[0]
    ymin1 = get_rotated_vertex(center, (right, bottom), np.deg2rad(angle))[1]
    xmax1 = get_rotated_vertex(center, (right, top), np.deg2rad(angle))[0]
    ymax1 = get_rotated_vertex(center, (left, top), np.deg2rad(angle))[1]

    # Compute dx and dy
    dx = (xmax1 - xmin1) / rot_ds.shape[2]
    dy = (ymax1 - ymin1) / rot_ds.shape[1]

    # Get extent of area of interest
    xmin2, ymin2, xmax2, ymax2 = extent

    # Compute bounds (in indices) corresponding to extent
    # Note that ymax1 and ymax2 are the actual coordinate values, therefore the ymin index is computed using ymax coordinates
    xmin = int((xmin2 - xmin1) / dx)
    xmax = int(rot_ds.shape[2] - (xmax1 - xmax2) / dx)
    ymin = int((ymax1 - ymax2) / dy)
    ymax = int(rot_ds.shape[1] - (ymin2 - ymin1) / dy)

    rot_ds = rot_ds[:, ymin:ymax, xmin:xmax]
    return rot_ds


def fit_array_to_bounds(array, array_bounds, center, extent, angle):
    """
    Crops the array to the extent. This assumes that array is rotated by angle degrees
    and extent is the extent after rotation. array bounds are the original bounds before rotation

    Input:
        - array, a 'rotated' array which we want to crop
        - array_bounds, the bounds or extent of the original array before rotation
        - center, the center point of the extent to which we want to crop
        - extent, the extent to which we want to crop
        - angle, the rotation angle of the array

    Output:
        - array, the cropped array
        - dx, the distance one cell in x direction represents
        - dy, the distance one cell in y direction represents
    """
    left, bottom, right, top = array_bounds

    # Compute new vertex positions after rotation
    xmin1 = get_rotated_vertex(center, (left, bottom), np.deg2rad(angle))[0]
    ymin1 = get_rotated_vertex(center, (right, bottom), np.deg2rad(angle))[1]
    xmax1 = get_rotated_vertex(center, (right, top), np.deg2rad(angle))[0]
    ymax1 = get_rotated_vertex(center, (left, top), np.deg2rad(angle))[1]

    # Get extent of area of interest
    xmin2, ymin2, xmax2, ymax2 = extent

    # Assumes that array dimensions are (y, x) or (z, y, x)
    if len(array.shape) == 2:
        # Compute dx and dy
        dx = np.round((xmax1 - xmin1) / array.shape[1])
        dy = np.round((ymax1 - ymin1) / array.shape[0])

        # Compute bounds (in indices) corresponding to extent
        # Note that ymax1 and ymax2 are the actual coordinate values, therefore the ymin index is computed using ymax coordinates
        xmin = int((xmin2 - xmin1) / dx)
        xmax = int(array.shape[1] - (xmax1 - xmax2) / dx)
        ymin = int((ymax1 - ymax2) / dy)
        ymax = int(array.shape[0] - (ymin2 - ymin1) / dy)

        array = array[ymin:ymax, xmin:xmax]
    elif len(array.shape) == 3:
        # Compute dx and dy
        dx = np.round((xmax1 - xmin1) / array.shape[2])
        dy = np.round((ymax1 - ymin1) / array.shape[1])

        # Compute bounds (in indices) corresponding to extent
        # Note that ymax1 and ymax2 are the actual coordinate values, therefore the ymin index is computed using ymax coordinates
        xmin = int((xmin2 - xmin1) / dx)
        xmax = int(array.shape[2] - (xmax1 - xmax2) / dx)
        ymin = int((ymax1 - ymax2) / dy)
        ymax = int(array.shape[1] - (ymin2 - ymin1) / dy)

        array = array[:, ymin:ymax, xmin:xmax]
    return (array, dx, dy)


def compute_mid_point_rectangle(rect_bounds):
    """
    Function for computing the center point of the rectangle defined by rect_bounds

    Input:
        -rect_bounds, the boundary vertices of the rectangle in the order of (left, bottom, right, top)

    Output:
        -the center point of the rectangle in (x, y) or (lat, lon)
    """

    x = (rect_bounds[0] + rect_bounds[2]) / 2
    y = (rect_bounds[1] + rect_bounds[3]) / 2

    return [x, y]


def get_frame_data(path):
    # Read tiff file and transpose bands to last dimension
    sat_tif = rxr.open_rasterio(path)
    sat_img = sat_tif.values
    sat_img = np.transpose(sat_img, (1, 2, 0))

    # Convert CRS to 28992 and get bounds
    sat_bounds = sat_tif.rio.reproject("EPSG:28992").rio.bounds()
    sat_extent = (sat_bounds[0], sat_bounds[2], sat_bounds[1], sat_bounds[3])

    # Get year of satellite image
    sat_text = path.stem[:4]

    return {"image": sat_img, "extent": sat_extent, "text": sat_text}
