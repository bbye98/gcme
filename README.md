# Gaussian core model with smeared electrostatics (GCMe)

This repository contains the code used in the journal article

> Ye, B. B.; Chen, S.; Wang, Z.-G. GCMe: Efficient Implementation of Gaussian Core Model with Smeared Electrostatic Interactions for Molecular Dynamics Simulations of Soft Matter Systems. **2024**. https://doi.org/10.48550/ARXIV.2403.08148.

### Pre-requisites

The Python scripts require

* [Python](https://www.python.org/downloads/)
3.9 or later,
* [OpenMM](
http://docs.openmm.org/latest/userguide/application/01_getting_started.html),
* [MDCraft](https://github.com/bbye98/mdcraft) and its dependencies, and
* either [`constvplugin`](https://github.com/scychon/openmm_constV) or
[`openmm-ic-plugin`](
https://github.com/bbye98/mdcraft/tree/main/lib/openmm-ic-plugin)

to be installed.

The LAMMPS scripts have been tested to run on the 21 Nov 2023 release
with the `fix imagecharges` command from [`lammps-fix-imagecharges`](
https://github.com/bbye98/mdcraft/tree/main/lib/lammps-fix-imagecharges).
Older LAMMPS builds will likely have to use the `fix imagecharges`
command from [`lammps-fixes`](https://github.com/kdwelle/lammps-fixes)
instead due to recent internal LAMMPS API changes.

### Directory

    ├── benchmark
    │   ├── ljcoul_ic_real.lmp      # LAMMPS: WCA/Coulomb system w/ image charges
    │   ├── ljcoul_slab_real.lmp    # LAMMPS: Slab WCA/Coulomb system
    │   ├── ljcoul_ic.py            # OpenMM: WCA/Coulomb system w/ image charges
    │   ├── gcme_bulk_real.lmp      # LAMMPS: Bulk GCMe system
    │   ├── gcme_ic_real.lmp        # LAMMPS: GCMe system w/ image charges
    │   ├── gcme_slab_real.lmp      # LAMMPS: Slab GCMe system
    │   └── gcme_all.py             # OpenMM: GCMe systems w/ all three BCs
    ├── analysis_gcme.ipynb
    ├── npt_water.py
    ├── nvt_polyanion_counterion_solvent.py
    └── nvt_water.py

The `benchmark` directory contains scripts to run simulations of simple
coarse-grained systems for the benchmark results in the "Performance"
section of the paper.

The `npt_water.py` script runs NpT simulations of coarse-grained "water"
particles at different pressures and repulsion parameters to determine
the key GCMe parametrization relationship in the "Parametrization"
section of the paper. The `nvt_water.py` script runs NVT simulations of
the parametrized GCM so that the most probable pair separation distance
can be determined using the radial distribution function.

The `nvt_polyanion_counterion_solvent.py` script runs NVT simulations of
polyanions, their counterions, and solvent particles confined between
two planar perfectly conducting or nonmetal electrodes using OpenMM as
part of the "Illustrative examples" section of the paper.

The `analysis_gcme.ipynb` Jupyter notebook analyzes and plots all
equations and simulation data included in the paper.