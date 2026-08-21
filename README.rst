===================
Virtual Climate Lab
===================

Virtual Climate Lab: An interactive visualization system for exploring geospatial environments with basemap and bathymetry data.

* Free software: GNU General Public License v3

Overview
--------

Virtual Climate Lab (VCL) is an interactive table-based visualization system designed for collaborative exploration of geospatial data. The system provides a framework for visualizing basemap imagery (aerial photos, satellite images) combined with bathymetry data, with support for additional custom data layers. It features intuitive multi-modal input controls including MIDI controllers, keyboard, hand tracking, and physical marker detection.

**Core Requirements:**

* **Basemap**: Aerial imagery or satellite basemap (required)
* **Bathymetry**: Depth/elevation data (required)

**Optional Data Layers:**

* Custom raster layers (PNG, GeoTIFF, NetCDF)
* Vector data (shapefiles, GeoPackage, GeoJSON)
* Particle simulations and flow fields
* Temporal/animated data sequences
* Statistical information panels

**Key Features:**

* **Multi-layer Geospatial Visualization**: Overlay custom data layers on basemap and bathymetry
* **Temporal Analysis**: Support for time-series and scenario data
* **Interactive Controls**: MIDI controller, keyboard, hand gesture tracking, and AprilTag/QR code detection
* **Real-time Animations**: Particle simulations and temporal progressions
* **Statistical Panels**: Layer-specific information displays with charts and infographics
* **Cross-section Views**: Vertical/horizontal slices through data for detailed analysis

Architecture
------------

VCL uses a multi-process architecture with ZeroMQ for inter-process communication:

**Display Components:**

* **DisplayMap**: Main map window showing geospatial layers and overlays (1920x1080)
* **DisplaySlice**: Cross-section viewer for vertical data slices
* **StatsWindow**: Statistics and information panels for each layer

**Input Handlers:**

* **Keyboard Publisher**: Translates keyboard input to layer commands
* **MIDI Board**: Processes MIDI controller input for intuitive layer control
* **Hand Tracker**: MediaPipe-based gesture recognition for touch-free interaction
* **UID Detector**: AprilTag/QR code/ArUco marker detection for physical interactivity

**Data Pipeline:**

1. **Preprocessing**: Converts raw geospatial data (PNG, TIF, NetCDF, shapefiles) to aligned rasters
2. **Caching**: Saves preprocessed data as pickle for fast loading
3. **Visualization**: Real-time rendering with Pygame and matplotlib

Installation
------------

**Install VCL using uv (recommended):**

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/openearth/vcl
    cd vcl
    
    # Install dependencies and sync environment
    uv sync

**Alternative installation with pip:**

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/openearth/vcl
    cd vcl
    
    # Create and activate virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    
    # Install dependencies
    pip install -e .

Quick Start
-----------

**1. Prepare Your Data:**

Create an ``input.json`` configuration file in the vcl package directory. **Required fields** are basemap, bathymetry, and extent. All other layers are optional:

.. code-block:: json

    {
        "basepath": "/path/to/data",
        "crs": "EPSG:4326",
        "common": {
            "basemap": "basemap.tif",           # Required: aerial/satellite image
            "bathymetry": "bathymetry.tif",     # Required: depth/elevation data
            "extent": "extent.geojson",         # Required: area of interest polygon
            "layers": {                          # Optional: custom data layers
                "layer1": "data1.png",
                "layer2": "data2.shp"
            },
            "stats": {},                         # Optional: statistical panels
            "animations": {},                    # Optional: temporal sequences
            "particles": {},                     # Optional: flow simulations
            "interactivity": {}                  # Optional: interactive regions
        }
    }

**2. Preprocess Data:**

.. code-block:: bash

    vcl --preprocess --save

**3. Launch the System:**

.. code-block:: bash

    # Full system with MIDI control
    vcl --midi --satellite --stats

    # With hand tracking and UID detection
    vcl --midi --satellite --stats --hand_tracking --uid

    # Keyboard-only mode
    vcl --no-midi --satellite

Museum Mode
-----------

For the museum kiosk setup, VCL can run in a simplified 2-process mode:

* **Process 1**: fixed input publisher for the five buttons
* **Process 2**: the single beamer display process

Museum mode disables the camera-based modules, hand tracking, UID detection,
stats window, slice window, and MIDI input. It also supports an inactivity
timeout and a slow automatic year loop.

**Direct startup command:**

.. code-block:: bash

    vcl --museum --default-layer overview --inactivity-timeout 120 --render-fps 30 --year-loop-fps 1

**Cross-platform startup wrapper from this repository:**

.. code-block:: bash

    python scripts/start_museum.py

On Windows:

.. code-block:: powershell

    py scripts\start_museum.py

The older shell wrapper remains available for macOS/Linux convenience:

.. code-block:: bash

    ./scripts/start_museum.sh

The startup wrapper reads these environment variables if you want to override
the defaults without editing the script:

* ``DEFAULT_LAYER``
* ``INACTIVITY_TIMEOUT``
* ``RENDER_FPS``
* ``YEAR_LOOP_FPS``

**Watchdog wrapper:**

.. code-block:: bash

    python scripts/watchdog_museum.py

On Windows:

.. code-block:: powershell

    py scripts\watchdog_museum.py

The watchdog restarts the museum runtime whenever it exits. By default it waits
5 seconds before restarting. You can change that with ``VCL_RESTART_DELAY``.

**Path handling:**

The Python startup wrapper is path-independent inside the repository. It resolves
the repository root from its own location and then looks for a Python interpreter
in this order:

* ``PYTHON_BIN`` if set
* the current Python executable
* ``.venv/bin/python``
* ``.venv/Scripts/python.exe``

OS-level service definitions still need an absolute path somewhere because that
is a requirement of the operating system service manager, not of VCL itself.
To avoid hand-editing those paths, generate the service files from this repo.

**macOS launchd setup:**

1. Generate the plist:

.. code-block:: bash

    python scripts/generate_museum_service.py launchd ~/Library/LaunchAgents/com.openearth.vcl.museum.plist

2. Load it:

.. code-block:: bash

    launchctl unload ~/Library/LaunchAgents/com.openearth.vcl.museum.plist 2>/dev/null || true
    launchctl load ~/Library/LaunchAgents/com.openearth.vcl.museum.plist

3. Inspect logs if needed:

.. code-block:: bash

    tail -f ~/Library/Logs/vcl-museum.stdout.log
    tail -f ~/Library/Logs/vcl-museum.stderr.log

The example file ``scripts/com.openearth.vcl.museum.plist.example`` is now a
placeholder template for reference.

**Linux systemd setup:**

1. Generate the unit:

.. code-block:: bash

    python scripts/generate_museum_service.py systemd ~/.config/systemd/user/vcl-museum.service

2. Enable and start it:

.. code-block:: bash

    systemctl --user daemon-reload
    systemctl --user enable vcl-museum.service
    systemctl --user start vcl-museum.service

3. Inspect logs:

.. code-block:: bash

    journalctl --user -u vcl-museum.service -f

The example file ``scripts/vcl-museum.service.example`` is also a placeholder
template for reference.

**Windows scheduled task setup:**

Use Task Scheduler to start the watchdog at login. A typical action is:

* Program/script: ``py``
* Add arguments: ``scripts\watchdog_museum.py``
* Start in: the repository root

If you prefer the full Python path instead of ``py``, point the task at the
interpreter inside ``.venv\Scripts\python.exe``.

For Windows there is no checked-in task XML yet, but the launcher and watchdog
are already cross-platform.

Command-Line Options
--------------------

.. code-block:: bash

    vcl [OPTIONS]

**Options:**

* ``--satellite / --no-satellite``: Enable main map display (default: False)
* ``--contour / --no-contour``: Enable cross-section viewer (default: False)
* ``--stats / --no-stats``: Enable statistics panels (default: False)
* ``--midi / --no-midi``: Use MIDI controller input (default: True)
* ``--hand_tracking / --no-hand_tracking``: Enable hand tracking (default: False)
* ``--uid / --no-uid``: Enable UID marker detection (default: False)
* ``--preprocess / --no-preprocess``: Run data preprocessing (default: False)
* ``--save / --no-save``: Save preprocessed data to disk (default: False)

Keyboard Controls
-----------------

When using keyboard mode (``--no-midi``), keyboard controls are available for layer navigation and interaction. Specific key mappings are configured in ``display_pygame.py`` and depend on your data setup.

**Common Functions:**

* ``LEFT`` / ``RIGHT``: Navigate slice position
* ``N`` / ``B``: Cycle next/previous in active layer collection
* ``M``: Mask layer toggle

**Note:** Layer selection keys and temporal controls are configured based on your specific data layers. See ``vcl/display_pygame.py`` in the ``keyboard_publisher()`` function for the complete key mapping configuration.

Data Formats
------------

VCL supports multiple geospatial data formats:

* **Raster**: PNG (with .pgw), GeoTIFF, NetCDF
* **Vector**: Shapefile, GeoPackage, GeoJSON
* **Particle**: NetCDF with mesh coordinates and velocity fields
* **Extent**: Any vector format defining the area of interest

All data is transformed to EPSG:4326 (WGS84) by default and aligned to the extent polygon.

Development
-----------

**Project Structure:**

.. code-block:: text

    vcl/
    ├── vcl/
    │   ├── cli.py              # Command-line interface
    │   ├── preprocess.py       # Data preprocessing pipeline
    │   ├── display_pygame.py   # Main display and input handlers
    │   ├── data.py            # Geospatial utilities
    │   ├── load_data.py       # Data loading
    │   ├── windows/           # Display window classes
    │   │   ├── DisplayMap.py
    │   │   ├── DisplaySlice.py
    │   │   └── StatsWindow.py
    │   ├── utils/             # Utility modules
    │   │   └── hand_tracking.py
    │   └── interactivity/     # UID detection
    │       └── uid_detection.py
    └── README.rst

Contributing
------------

Contributions are welcome! Please feel free to submit a Pull Request.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

Developed by Deltares and partners for interactive coastal zone management and climate scenario exploration.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

License
-------

GNU General Public License v3
