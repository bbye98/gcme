#!/usr/bin/env python3

"""
npt_water.py
============
Simulates isothermal–isobaric (NpT) systems of equisized particles
interacting via the nonelectrostatic Gaussian core model (GCM) pairwise
potential using the OpenMM. The resulting time-averaged volumes (or
number densities) from simulations with different pressures and GCM
repulsion parameters can be used to determine the scaling parameter in
the water GCM.
"""

import logging
import os
import platform
import sys

import numpy as np
import openmm
from openmm import app, unit
from scipy import optimize

from mdcraft.openmm import pair, system as s, topology as t, unit as u

ORIG_PATH = os.getcwd()                             # Original directory
ROOM_TEMP = 300 * unit.kelvin                       # Room temperature (300 K)
RHO = 0.99657 * unit.gram / unit.centimeter ** 3    # Water mass density (300 K)
MW = 18.01528 * unit.amu                            # Water molar mass
DIAMETER = 0.275 * unit.nanometer                   # Water molecule size

def run(N: int, p_md: float, A_md: float, frames: int, *,
        temperature: unit.Quantity = ROOM_TEMP, size: unit.Quantity = DIAMETER,
        mass: unit.Quantity = MW, freq: int = 100, u_shift_md: float = 1e-6,
        dt_md: float = 0.01, every: int = 10_000, device: int = 0,
        path: str = None, verbose: bool = True) -> None:

    """
    Run a NpT simulation of coarse-grained particles interacting via the
    GCM using the OpenMM CUDA platform.

    Parameters
    ----------
    N : `int`
        Total number of particles.

    p_md : `float`
        Reduced system pressure.

    A_md : `float`
        Reduced GCM repulsion parameter.

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

    freq : `int`, keyword-only, default: :code:`100`
        Monte Carlo pressure change attempt frequency, in timesteps.

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
        "length": size,
        "mass": mass
    })
    logging.info("Computed scaling factors for reducing physical quantities.\n"
                 "  Fundamental quantities:\n"
                 f"    Molar energy: {scales['molar_energy']}\n"
                 f"    Length: {scales['length']}\n"
                 f"    Mass: {scales['mass']}")

    # Determine the system dimensions
    rho = unit.AVOGADRO_CONSTANT_NA * RHO / scales["mass"]
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
        system, topology, N, scales["mass"],
        name="H2O",
        cnbforces={pair_gauss: (scales["length"] / 2,)}
    )
    logging.info(f"Registered {N:,} water particles to the simulation.")

    # Determine the filename prefix
    fname = f"npt_A_{A_md:.3f}_p_{p_md:.2f}"

    # Ensure a simulation with the same filename does not already exist
    if os.path.isfile(f"{fname}.log"):
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

    # Add Monte Carlo barostat and reinitialize simulation context
    pres = p_md * scales["pressure"]
    system.addForce(openmm.MonteCarloBarostat(pres, temperature, freq))
    logging.info(f"Registered Monte Carlo barostat at {pres} and "
                 f"{temperature} to the simulation.")
    simulation.context.reinitialize()
    simulation.context.setPositions(pos)

    # Write topology file
    with open(f"{fname}.cif", "w") as f:
        app.PDBxFile.writeFile(simulation.topology, pos, f, keepIds=True)
    logging.info(f"Wrote topology to '{fname}.cif'.")

    # Register checkpoint and thermodynamic state data reporters
    simulation.reporters.append(
        app.CheckpointReporter(f"{fname}.chk", 100 * every)
    )
    logging.info("Registered checkpoint reporter writing to "
                 f"'{fname}.cif' to the simulation.")
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
    logging.info("Registered state data reporter writing to "
                 f"'{fname}.log' to the simulation.")

    # Run NpT simulation
    logging.info(f"Starting NpT run with {timesteps:,} timesteps...")
    simulation.step(timesteps)
    simulation.saveState(f"{fname}.xml")
    logging.info("Simulation completed. Wrote final simulation state "
                 f"to '{fname}.xml'.")

if __name__ == "__main__":

    path: str = "/mnt/e/research/gcme/data/parametrization/water/npt"
    N: int = 10_000
    frames: int = 5_000

    As_md: np.ndarray = np.array((5, 6, 10, 20), dtype=float)
    ps_md: np.ndarray = np.array(
        (4_500, 4_000, 3_000, 2_000, 1_000, 750, 500, 400, 300, 200, 100, 75,
         50, 40, 30, 20, 10, 7.5, 5, 4, 3, 2, 1, 0.5),
        dtype=float
    )

    for p_md in ps_md:
        for A_md in As_md:
            run(N, p_md, A_md, frames, path=path)