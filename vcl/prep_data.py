from pathlib import Path

import numpy as np
import shapely

import vcl.data

data_dir = data_dir = Path("~/data/vcl/gnsbi").expanduser()


def preprocess_common(datasets):
    # Get bathymetry datasets
    ds_b0 = datasets["ds_b0"]
    # Create dictionary to store processed data and values
    preprocessed = {}

    # Get the extent we want to show
    preprocessed["extent"] = datasets[f"extent"]

    # Add satellite image
    sat = datasets["sat"]

    # Compute rotation angle of the extent as well as centre point of the extent
    preprocessed["angle"] = vcl.data.compute_rotation_angle(preprocessed["extent"])
    preprocessed["mid_point"] = preprocessed["extent"].centroid.coords[0]

    # Create shaded image from satellite and bathymetry
    preprocessed["sat"] = vcl.data.create_shaded_image(sat, ds_b0)
    print(preprocessed["sat"][..., -1])
    # Compute combined bounds of satellite image and the bathymetry
    preprocessed["plt_extent"] = vcl.data.sat_and_bodem_bounds(sat, ds_b0)
    # Get plot lims of rotated extent (rotated such that extent has an angle of 0 with the horizontal axis)
    preprocessed["plt_lims"] = vcl.data.get_plot_lims(
        shapely.affinity.rotate(preprocessed["extent"], -preprocessed["angle"])
    )

    # Set new extent
    preprocessed["sat_extent"] = vcl.data.sat_and_bodem_bounds(sat, ds_b0)

    # datasets["sat"] = (sat.read() * 255).astype(np.uint8)
    # datasets["sat"] = np.transpose(datasets["sat"], (1, 2, 0))

    sat.close()
    # GSR.close()

    return preprocessed


def preprocess_unique(datasets):
    preprocessed_datasets = {}
    for year in datasets.keys():
        preprocessed_datasets[year] = {}

    return preprocessed_datasets


def preprocess(common_datasets, unique_datasets):
    preprocessed_common_datasets = preprocess_common(common_datasets)
    preprocessed_unique_datasets = preprocess_unique(unique_datasets)

    # Combine common and unique preprocessed datasets for each year
    preprocessed_datasets = {}
    for year in unique_datasets.keys():
        preprocessed_datasets[year] = {
            **preprocessed_common_datasets,
            **preprocessed_unique_datasets[year],
        }
        # for scenario in unique_datasets[year]["ssp"].keys():
        #     preprocessed_datasets[year][f"ssp_{scenario}"] = np.load(
        #         data_dir / f"preprocessed-{year}-ssp-{scenario}.npy", allow_pickle=True
        #     ).item()
    return preprocessed_datasets
