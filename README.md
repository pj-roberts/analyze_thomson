# analyze_thomson: Plasma Diagnostic Analysis Interface

Copyright Parker J. Roberts, University of Michigan 2024 (?)

- The goal of this project is to be a library of python scripts for low-level analysis of incoherent Thomson scattering data for the low-temperature plasma experiments at the Plasmadynamics and Electric Propulsion Laboratory (PEPL) at the University of Michigan.
- This library includes code for:
    - Parsing of raw data from detectors,
    - Accumulation of laser shots and composing of full scattering spectra, 
    - Calibration analysis from Raman scattering data, and ...
    - Processing of spectra using model fits to obtain plasma properties.
- We also are moving toward the creation of a user interface using "tkinter" for ease of code-free analysis by non-experts.

- The philosophical goals of this software library are:
    - Generate clean, publication-ready figures from analysis
    - Save intermediate processing data structures for modularity
    - Maintain code organization that allows readability and mutability

- Organizational structure:
    - Each directory, including the root .../src, begins with 3 modules:
        - constants.py
        - functions.py
        - main.py, which may be named in accordance with the package
    - As complexity grows, related functions are grouped as classes and packages.
