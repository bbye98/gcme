#!/usr/bin/env python3

"""
nvt_water.py
============
Simulates canonical (NVT) systems of equisized particles interacting via
the nonelectrostatic Gaussian core model (GCM) pairwise potential using
the OpenMM. The resulting trajectories can be used to compute structural
properties, such as the radial distribution function to determine the
most probable distance between two particles.
"""

import logging
import os
import platform
import sys
import warnings

import numpy as np
import openmm
from openmm import app, unit
from scipy import optimize

from mdcraft.openmm import pair, reporter, system as s, topology as t, unit as u

ORIG_PATH = os.getcwd()                             # Original directory
ROOM_TEMP = 300 * unit.kelvin                       # Room temperature (300 K)
MW = 18.01528 * unit.amu                            # Water molar mass
DIAMETER = 0.275 * unit.nanometer                   # Water molecule size
KAPPA_INV = 15.9835                                 # Water dimensionless compressibility (300 K)
OMEGA = 0.499                                       # GCM scaling parameter

def run(N: int, frames: int, *, temperature: unit.Quantity = ROOM_TEMP,
        size: unit.Quantity = DIAMETER, mass: unit.Quantity = MW, N_m: int = 4,
        rho_md: float = 2.5, u_shift_md: float = 1e-6, dt_md: float = 0.01,
        every: int = 10000, device: int = 0, path: str = None,
        verbose: bool = True) -> None:

    """
    Run a NVT simulation of coarse-grained particles interacting via the
    GCM using the OpenMM CUDA platform.

    Parameters
    ----------
    N : `int`
        Total number of particles.

    frames : `int`
        Total number of frames. (The total number of timesteps is
        `frames` times `every`.)

        **Example**: :code:`3000` for 3,000 frames, or a total of
        :code:`3000 * every` timesteps.

    temperature : `openmm.unit.Quantity`, keyword-only, \
    default: :code:`300 * unit.kelvin`
        System temperature.

        **Reference unit**: :math:`\\mathrm{K}`.

    size : `openmm.unit.Quantity`, keyword-only, \
    default: :code:`0.275 * unit.nanometer`
        Simulation bead diameter.

        **Reference unit**: :math:`\\mathrm{nm}`.

    mass : `openmm.unit.Quantity`, keyword-only, \
    default: :code:`18.01528 * unit.amu`
        Simulation bead mass.

        **Reference unit**: :math:`\\mathrm{g/mol}`.

    N_m : `float`, keyword-only, default: :code:`3.0`
        GCM real space scaling parameter, roughly defined as the number
        of water molecules per simulation bead.

    rho_md : `float`, keyword-only, default: :code:`2.5`
        Reduced number density.

    u_shift_md : `float`, keyword-only, default: :code:`1e-6`
        Reduced potential energy at which to truncate and shift the
        nonelectrostatic excluded volume interactions.

    dt_md : `float`, keyword-only, default: :code:`0.01`
        Reduced timestep.

    every : `int`, keyword-only, default: :code:`10000`
        Thermodynamic data and trajectory output frequency.

    device : `int`, keyword-only, default: :code:`0`
        CUDA device index.

        **Valid values**: `device` must be greater than or equal to 0.

    path : `str`, keyword-only, optional
        Directory to store data. If it does not exist, it is created. If
        not specified, the directory containing this script is used.

    verbose : `bool`, keyword-only, default: :code:`True`
        Determines whether detailed progress is shown.
    """

    # Set up logger
    logging.basicConfig(format="{asctime} | {levelname:^8s} | {message}",
                        style="{",
                        level=logging.INFO if verbose else logging.WARNING)

    # Change to the data directory
    if path is None:
        path = ORIG_PATH
    if not os.path.isdir(path):
        os.makedirs(path)
        logging.info(f"Created data directory '{path}'.")
    os.chdir(path)
    logging.info(f"Changed to data directory '{path}'.")

    # Determine the parameter scales using the fundamental quantities
    scales = u.get_lj_scaling_factors({
        "energy": (unit.BOLTZMANN_CONSTANT_kB
                   * temperature).in_units_of(unit.kilojoule),
        "length": size * (N_m * rho_md) ** (1 / 3) if N_m > 1 else size,
        "mass": mass * N_m
    })
    logging.info("Computed scaling factors for reducing physical quantities.\n"
                 "  Fundamental quantities:\n"
                 f"    Molar energy: {scales['molar_energy']}\n"
                 f"    Length: {scales['length']}\n"
                 f"    Mass: {scales['mass']}")

    # Determine the system dimensions
    rho = rho_md / scales["length"] ** 3
    L_nd = (N / rho) ** (1 / 3) / unit.nanometer
    dims = np.array((L_nd, L_nd, L_nd)) * unit.nanometer

    # Initialize simulation system and topology
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(*(dims * np.diag(np.ones(3))))
    topology = app.Topology()
    topology.setUnitCellDimensions(dims)
    logging.info("Created simulation system and topology with "
                 f"dimensions {dims[0]} x {dims[1]} x {dims[2]}.")

    # Set up the nonelectrostatic excluded volume interactions
    if rho_md <= 2.3:
        wmsg = ("The water Gaussian core model parameters should only "
                "be used at sufficiently high reduced number densities "
                "(ρ* > 2.3).")
        warnings.warn(wmsg)
    A_md = (N_m * KAPPA_INV - 1) / (2 * OMEGA * rho_md)
    A = A_md * scales["molar_energy"] * scales["length"] ** 3
    beta = 3 / scales["length"] ** 2
    cutoff = optimize.fsolve(
        lambda r: A * (beta / np.pi) ** (3 / 2)
                  * np.exp(-beta * (r * unit.nanometer) ** 2)
                  / scales["molar_energy"] - u_shift_md,
        scales["length"].value_in_unit(unit.nanometer)
    )[0] * unit.nanometer
    pair_gauss = pair.gauss(cutoff, mix="core", global_params={"A": A})

    # Register force field to simulation
    system.addForce(pair_gauss)
    logging.info(f"Registered {system.getNumForces()} pair "
                 "potential(s) to the simulation.")

    # Register particles to pair potential
    s.register_particles(
        system, topology, N, scales["mass"], name="H2O", resname="H2O",
        cnbforces={pair_gauss: (scales["length"] / 2,)}
    )
    logging.info(f"Registered {N:,} water particles to the simulation.")

    # Determine the filename prefix
    fname = f"nvt_A_{A_md:.3f}_Nm_{N_m}_rho_{rho_md:.2f}"

    # Ensure a simulation with the same filename does not already exist
    if os.path.isfile(f"{fname}.nc"):
        emsg = (f"A simulation with the filename prefix '{fname}' "
                "already exists.")
        raise RuntimeError(emsg)

    # Create OpenMM CUDA Platform
    plat = openmm.Platform.getPlatformByName("CUDA")
    properties = {"Precision": "mixed", "DeviceIndex": str(device),
                  "UseBlockingSync": "false"}
    dt = dt_md * scales["time"]
    fric = 1e-3 / dt
    logging.info(f"Initialized the {plat.getName()} platform in OpenMM "
                 f"{plat.getOpenMMVersion()} on {platform.node()}.")

    # Generate initial particle positions
    pos = t.create_atoms(dims, N)
    logging.info("Generated random initial configuration for "
                 f"{len(pos):,} particles.")

    # Perform NVT energy minimization
    logging.info("Starting system relaxation...")
    integrator = openmm.LangevinMiddleIntegrator(temperature, fric, dt)
    simulation = app.Simulation(topology, system, integrator, plat,
                                properties)
    simulation.context.setPositions(pos)
    simulation.minimizeEnergy()
    pos = simulation.context.getState(getPositions=True).getPositions(
        asNumpy=True
    )
    logging.info("Local energy minimization completed.")

    # Write topology file
    with open(f"{fname}.cif", "w") as f:
        app.PDBxFile.writeFile(simulation.topology, pos, f, keepIds=True)
    logging.info(f"Wrote topology to '{fname}.cif'.")

    # Register checkpoint, thermodynamic state data, and trajectory reporters
    simulation.reporters.append(
        app.CheckpointReporter(f"{fname}.chk", 100 * every)
    )
    logging.info("Registered checkpoint reporter writing to "
                 f"'{fname}.cif' to the simulation.")
    simulation.reporters.append(reporter.NetCDFReporter(f"{fname}.nc", every))
    logging.info("Registered trajectory reporter writing to "
                 f"'{fname}.nc' to the simulation.")
    timesteps = frames * every
    for o in [sys.stdout, f"{fname}.log"]:
        simulation.reporters.append(
            app.StateDataReporter(
                o, reportInterval=every, step=True, temperature=True,
                volume=True, potentialEnergy=True, kineticEnergy=True,
                totalEnergy=True, remainingTime=True, speed=True,
                totalSteps=timesteps
            )
        )
    logging.info("Registered state data reporter that will write to "
                 f"'{fname}.log' to the simulation.")

    # Run NVT simulation
    logging.info(f"Starting NVT run with {timesteps:,} timesteps...")
    simulation.step(timesteps)
    simulation.saveState(f"{fname}.xml")
    logging.info("Simulation completed. Wrote final simulation state "
                 f"to '{fname}.xml'.")

if __name__ == "__main__":

    path: str = "/mnt/e/research/gcme/data/parametrization/water/nvt"
    N: int = 10_000
    frames: int = 5_000

    run(N, frames, N_m=4, path=path)