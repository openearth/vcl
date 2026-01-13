"""
Virtual Climate Lab Windows Package.

This package provides pygame and matplotlib-based window classes for interactive
visualization of climate and oceanographic data. The windows support features
including multiple data layers, zooming, panning, animations, and statistical plots.

Modules:
    PygameWindow: Base class for pygame-based visualization windows.
    DisplayMap: Interactive map display with zoom, pan, flow visualization, and animations.
    DisplaySlice: 2D cross-sectional slice visualization from 3D datasets.
    StatsWindow: Statistics and plot display using matplotlib.

Usage Example:
    from vcl.windows import DisplayMap, StatsWindow

    # Create a map window
    map_window = DisplayMap(datasets, flow_data, dataset_kwargs, bg_layer='bathymetry')

    # Create a stats window
    stats_window = StatsWindow(stats_data, stats_kwargs)

    # Run the windows
    while True:
        map_window.draw_layers()
        stats_window.draw_stats()
"""
