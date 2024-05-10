#!/usr/bin/env python3

"""
nvt_polyanion_cation_wall.py
============================
Simulates canonical (NVT) systems of equisized polyanions, their
counterions, and solvent particles confined between two planar
electrodes using OpenMM. The force field used is the Gaussian core model
with smeared electrostatic interactions (GCMe), and the boundary
polarizability is accounted for using either the method of image charges
or the Yeh–Berkowitz slab correction.
"""

from itertools import combinations
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

ORIG_PATH = os.getcwd()             # Original directory
ROOM_TEMP = 300 * unit.kelvin       # Room temperature (300 K)
MW = 18.01528 * unit.amu            # Water molar mass
DIAMETER = 0.275 * unit.nanometer   # Water molecule size
KAPPA_INV = 15.9835                 # Water dimensionless compressibility (300 K)
OMEGA = 0.499                       # GCM scaling parameter

def run(N: int, N_p: int, x_p: float, bc: str, frames: int, *,
        temperature: unit.Quantity = ROOM_TEMP, size: unit.Quantity = DIAMETER,
        mass: unit.Quantity = MW, N_m: float = 4.0, varepsilon_r: float = 78.0,
        rho_md: float = 2.5, b_md: float = 0.8, k_md: float = 100.0,
        u_shift_md: float = 1e-3, dt_md: float = 0.02,  every: int = 10_000,
        a_scale: float = 1.0, L_z_scale: float = 2.5, device: int = 0,
        index: int = None, path: str = None, verbose: bool = True) -> None:

    """
    Run a NVT simulation of polyanions, counterions, and solvent
    particles confined between two planar walls using OpenMM.

    Parameters
    ----------
    N : `int`
        Total number of particles.

    N_p : `int`
        Polyanion chain length.

    x_p : `float`
        Polyanion number concentration.

        **Valid values**: `x_p` must be between 0 and 0.5.

    bc : `str`
        Boundary condition.

        .. container::

           **Valid values**:

           * :code:`"ic"` for the method of image charges.
           * :code:`"slab"` for the Yeh–Berkowitz slab correction.

    frames : `int`
        Total number of frames. (The total number of timesteps is
        `frames` times `every`.)

        **Example**: :code:`3_000` for 3,000 frames, or a total of
        :code:`3_000 * every` timesteps.

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

    varepsilon_r : `float`, keyword-only, default: :code:`78.0`
        Relative permittivity.

    rho_md : `float`, keyword-only, default: :code:`2.5`
        Reduced number density.

    b_md : `float`, keyword-only, default: :code:`0.8`
        Reduced equilibrium bond length.

    k_md : `float`, keyword-only, default: :code:`100.0`
        Reduced bond spring constant.

    u_shift_md : `float`, keyword-only, default: :code:`1e-3`
        Reduced potential energy at which to truncate and shift the
        nonelectrostatic excluded volume interactions.

    dt_md : `float`, keyword-only, default: :code:`0.02`
        Reduced timestep.

    every : `int`, keyword-only, default: :code:`10000`
        Thermodynamic data and trajectory output frequency.

    a_scale : `float`, keyword-only, default: :code:`1.0`
        Ratio of the electrostatic and mass smearing radii.

    L_z_scale : `float`, keyword-only, default: :code:`2.5`
        Approximate ratio of the system z-dimension to the x- and y-
        dimensions.

    device : `int`, keyword-only, default: :code:`0`
        CUDA device index.

        **Valid values**: `device` must be greater than or equal to 0.

    index : `int`, keyword-only, optional
        Simulation run index.

        **Valid values**: `index` must be greater than or equal to 0.

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
        "energy": (unit.BOLTZMANN_CONSTANT_kB * temperature).in_units_of(
            unit.kilojoule
        ),
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
    L_nd = ((N / (L_z_scale * rho)) ** (1 / 3)).value_in_unit(unit.nanometer)
    pos_wall, dims = t.create_atoms(np.array((L_nd, L_nd, 0)),
                                    lattice="hcp",
                                    length=scales["length"] / 2,
                                    flexible=True)
    dims[2] = N / (rho * dims[0] * dims[1])

    # Initialize simulation system and topology
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        *(dims * np.diag(np.ones(3, dtype=float)))
    )
    topology = app.Topology()
    topology.setUnitCellDimensions(dims)
    logging.info("Created simulation system and topology with "
                 f"dimensions {dims[0]} x {dims[1]} x {dims[2]}.")

    # Set up the nonelectrostatic excluded volume interactions
    # Types: real particle (p), wall (w), image charge (i)
    if rho_md <= 2.3:
        wmsg = ("The Gaussian core model parameters should only be "
                "used at sufficiently high reduced number densities "
                "(ρ* > 2.3).")
        warnings.warn(wmsg)
    radius_nd = scales["length"].value_in_unit(unit.nanometer) / 2
    sigmas_i_sq = (np.array((radius_nd, 0, radius_nd)) * unit.nanometer) ** 2
    sigmas_ij_sq = sigmas_i_sq + sigmas_i_sq[:, None]
    betas_ij = 3 / (2 * sigmas_ij_sq)
    alphas_ij_coefs = 1 + np.array((
        (0, 0, -1),     # pp, pw, pi;
        (0, -1, -1),    # wp, ww, wi;
        (-1, -1, -1)    # ip, iw, ii
    ))
    A_md = (N_m * KAPPA_INV - 1) / (2 * OMEGA * rho_md)
    A = A_md * scales["molar_energy"] * scales["length"] ** 3
    alphas_ij = alphas_ij_coefs * A * (betas_ij / np.pi) ** (3 / 2)
    alphas_ij[np.isnan(alphas_ij)] = 0 * unit.kilojoule_per_mole
    cutoff = optimize.fsolve(
        lambda r: np.max(alphas_ij)
                  * np.exp(-np.min(betas_ij) * (r * unit.nanometer) ** 2)
                  / scales["molar_energy"] - u_shift_md,
        scales["length"].value_in_unit(unit.nanometer)
    )[0] * unit.nanometer
    pair_gauss = pair.gauss(cutoff, mix="alpha12=alpha(type1,type2);"
                                        "beta12=beta(type1,type2);",
                            per_params=("type",),
                            tab_funcs={"alpha": alphas_ij, "beta": betas_ij})

    # Set up the electrostatic smeared Coulomb potential
    # Types: real or image particle (p), wall (w)
    as_i_sq = (np.array((a_scale, 0)) * scales["length"] / 2) ** 2
    as_ij = (as_i_sq + as_i_sq[:, None]) ** (1 / 2)
        # pp, pw; wp, ww
    e = 1 * unit.elementary_charge
    dielectric_min = np.ceil(unit.AVOGADRO_CONSTANT_NA * e ** 2
                             * (np.pi * np.max(sigmas_ij_sq) ** 3
                                / 27) ** (1 / 2)
                             / (u.VACUUM_PERMITTIVITY * A * as_ij[0, 0]))
    if varepsilon_r < dielectric_min:
        wmsg = (f"The relative permittivity ε={varepsilon_r} is too "
                "low, which can cause oppositely-charged ions to "
                "collapse onto each other. The minimum value is "
                f"approximately ε={dielectric_min:.6g}.")
        warnings.warn(wmsg)
    q_scaled = e / np.sqrt(varepsilon_r)
    pair_elec_dir, pair_elec_rec = pair.coul_gauss(
        cutoff, mix="alpha12=alpha(type1,type2);", per_params=("type",),
        tab_funcs={"alpha": np.sqrt(np.pi / 2) / as_ij}
    )

    # Set up the harmonic bond potential
    if x_p > 0:
        b = b_md * scales["length"]
        k = k_md * scales["molar_energy"] / scales["length"] ** 2
        bond_harm = openmm.HarmonicBondForce()

    # Register force field to simulation
    system.addForce(pair_gauss)
    system.addForce(pair_elec_dir)
    system.addForce(pair_elec_rec)
    if x_p > 0:
        system.addForce(bond_harm)
    logging.info(f"Registered {system.getNumForces()} pair "
                 "potential(s) to the simulation.")

    # Assign arbitrary particle identities
    element_a = app.Element.getBySymbol("Cl")
    element_c = app.Element.getBySymbol("Na")
    element_s = app.Element.getBySymbol("Ar")
    element_w = app.Element.getBySymbol("C")

    # Determine the number of polyanions, counterions, and solvent particles
    M = round(x_p * N / N_p)    # Number of polyanions
    N_a = N_c = M * N_p         # Number of polyanion beads and/or counterions
    N_s = N - N_a - N_c         # Number of solvent particles
    if N_a != x_p * N:
        emsg = (f"The polyanion chain length {N_p=} is incompatible "
                f"with the total number of particles {N=} and the "
                f"polyanion number concentration {x_p=}.")
        raise RuntimeError(emsg)

    # Register polyanions to pair potentials
    for _ in range(M):
        chain = topology.addChain()
        s.register_particles(
            system, topology, N_p, scales["mass"], chain=chain,
            element=element_a, name="PAN", resname="PAN",
            nbforce=pair_elec_rec, charge=-q_scaled,
            cnbforces={pair_elec_dir: (-q_scaled, 0), pair_gauss: (0,)}
        )
    logging.info(f"Registered {M:,} polyanion(s) with {N_p:,} monomer(s) "
                 "to the force field.")

    # Register polyanion bonds to bond potential and remove 1-2 interactions
    if x_p > 0:
        atoms = list(topology.atoms())
        for m in range(M):
            for n in range(N_p - 1):
                i = m * N_p + n
                j = i + 1
                topology.addBond(atoms[i], atoms[j])
                bond_harm.addBond(i, j, b, k)
                pair_elec_dir.addExclusion(i, j)
                pair_elec_rec.addException(i, j, 0, 0, 0)
                pair_gauss.addExclusion(i, j)
        logging.info(f"Registered {topology.getNumBonds():,} bond(s) to "
                     "the force field.")

    # Register counterions to pair potentials
    s.register_particles(
        system, topology, N_c, scales["mass"],
        element=element_c, name="CAT", resname="CAT",
        nbforce=pair_elec_rec, charge=q_scaled,
        cnbforces={pair_elec_dir: (q_scaled, 0), pair_gauss: (0,)}
    )
    logging.info(f"Registered {N_c:,} counterion(s) to the simulation.")

    # Register solvent particles to pair potentials
    s.register_particles(
        system, topology, N_s, scales["mass"],
        element=element_s, name="SOL", resname="SOL",
        nbforce=pair_elec_rec,
        cnbforces={pair_elec_dir: (0, 0), pair_gauss: (0,)}
    )
    logging.info(f"Registered {N_s:,} solvent particle(s) to the simulation.")

    # Determine positions and number of wall particles
    pos_wall = np.concatenate((
        pos_wall,
        pos_wall + np.array(
            (0, 0, dims[2].value_in_unit(unit.nanometer))
        ) * unit.nanometer
    ))
    N_w = pos_wall.shape[0]

    # Register wall particles to pair potentials
    for name in ("LWL", "RWL"):
        s.register_particles(
            system, topology, N_w // 2, 0,
            element=element_w, name=name,
            nbforce=pair_elec_rec,
            cnbforces={pair_elec_dir: (0, 1), pair_gauss: (1,)}
            # cnbforces={pair_elec_dir: (0, 1), pair_gauss: (1,)}
        )
    logging.info(f"Registered {N_w:,} wall particles to the force field.")

    # Remove wall–wall interactions
    wall_indices = range(N, N + N_w)
    for i, j in combinations(wall_indices, 2):
        pair_elec_dir.addExclusion(i, j)
        pair_elec_rec.addException(i, j, 0, 0, 0)
        pair_gauss.addExclusion(i, j)
    logging.info("Removed wall–wall interactions.")

    # Determine the filename prefix
    if index is None:
        index = 0
    fname = (f"nvt_N_{N}_Np_{N_p}_xp_{x_p:.3f}_rp_{varepsilon_r:.1f}_"
             f"A_{A_md:.3f}__{index}")

    # Ensure a simulation with the same filename does not already exist
    if os.path.isfile(f"{fname}.nc"):
        emsg = (f"A simulation with the filename prefix '{fname}' "
                "already exists.")
        raise RuntimeError(emsg)

    # Check for a previous run to continue from
    if index:
        prev_fname = (f"nvt_N_{N}_Np_{N_p}_xp_{x_p:.3f}_rp_"
                      f"{varepsilon_r:.1f}_A_{A_md:.3f}__{index - 1}")
    else:
        prev_fname = None
    if prev_fname and not os.path.isfile(f"{prev_fname}.chk") \
            and not os.path.isfile(f"{prev_fname}.xml"):
        prev_fname = None

    # Create OpenMM CUDA Platform
    plat = openmm.Platform.getPlatformByName("CUDA")
    properties = {"Precision": "mixed", "DeviceIndex": str(device),
                  "UseBlockingSync": "false"}
    dt = dt_md * scales["time"]
    fric = 1e-3 / dt
    logging.info(f"Initialized the {plat.getName()} platform in OpenMM "
                 f"{plat.getOpenMMVersion()} on {platform.node()}.")

    # Set up simulation system
    while True:

        # Generate initial particle positions
        if x_p == 1:
            pos_sys = t.create_atoms(dims, N_a, N_p=N_p, length=b,
                                     randomize=True)
        elif x_p == 0:
            pos_sys = t.create_atoms(dims, N_s)
        else:
            pos_sys = np.concatenate((
                t.create_atoms(dims, N_a, N_p=N_p, length=b, randomize=True),
                t.create_atoms(dims, N_c + N_s)
            )) * unit.nanometer

        # Scale z-positions to prevent particles from clipping through the walls
        pos_z_factor = 0.1
        pos_z_wall = 2 ** (-5 / 6) * scales["length"]
        pos_z_scale = (((1 - pos_z_factor) * dims[2] - pos_z_wall)
                       / (np.max(pos_sys[:, 2]) - np.min(pos_sys[:, 2])))
        pos_sys *= np.array((1, 1, pos_z_scale))
        pos_sys[:, 2] += (pos_z_wall + pos_z_factor * dims[2]) / 2 \
                         - pos_z_scale * np.min(pos_sys[:, 2])

        # Concatenate particle and wall positions
        pos = np.concatenate((pos_sys, pos_wall)) * unit.nanometer
        if prev_fname:
            break
        logging.info("Generated random initial configuration for "
                     f"{N:,} particles and {N_w:,} wall particles.")

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
        if pos[:N, 2].min() > 0 * unit.nanometer and pos[:N, 2].max() < dims[2]:
            logging.info("Local energy minimization completed.")
            break
        logging.warning("Particles have escaped the simulation box! "
                        "Trying again...")

    # Apply method of image charges or slab correction
    if bc == "ic":
        pos, integrator = s.add_image_charges(
            system, topology, pos, temperature, fric, dt,
            wall_indices=wall_indices,
            nbforce=pair_elec_rec, cnbforces={
                pair_elec_dir: {"charge": 0}, pair_gauss: {"replace": {0: 2}}
            }
        )
    elif bc == "slab":
        integrator = s.add_slab_correction(system, topology, pair_elec_rec,
                                           temperature, fric, dt)

    # Set up new simulation system with the updated integrator
    simulation = app.Simulation(topology, system, integrator, plat, properties)
    if prev_fname:
        try:
            simulation.loadCheckpoint(f"{prev_fname}.chk")
            logging.info("Previous simulation state loaded from "
                         f"'{prev_fname}.chk'.")
        except:
            simulation.loadState(f"{prev_fname}.xml")
            logging.info(f"Previous simulation state loaded from "
                         f"'{prev_fname}.xml'.")
        pos = simulation.context.getState(getPositions=True).getPositions(
            asNumpy=True
        )
    else:
        simulation.context.setPositions(pos)
        vel_ref = np.sqrt(
            3 * scales["molar_energy"] / scales["mass"]
        ).in_units_of(unit.nanometer / unit.picosecond)
        vel = np.random.uniform(-1, 1, (N, 2))
        vel /= np.linalg.norm(vel, axis=1)[:, None]
        vel = np.hstack((vel, np.zeros((N, 1))))
        vel = np.vstack((vel, np.zeros((N_w, 3))))
        if bc == "ic":
            vel = np.vstack((vel, np.zeros_like(vel)))
        vel *= vel_ref
        simulation.context.setVelocities(vel)

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
    logging.info("Registered state data reporter writing to "
                 f"'{fname}.log' to the simulation.")

    # Run NVT simulation
    logging.info(f"Starting NVT run with {timesteps:,} timesteps...")
    simulation.step(timesteps)
    simulation.saveState(f"{fname}.xml")
    logging.info("Simulation completed. Wrote final simulation state "
                 f"to '{fname}.xml'.")

if __name__ == "__main__":

    path: str = "/mnt/e/research/gcme/data/polyanion_counterion_solvent/edl"
    N: int = 96_000
    N_p: int = 60
    frames: int = 1_100

    xs_p: list[float] = [0.05, 0.025, 0.1, 0.005, 0.2]
    bcs: list[str] = ["ic", "slab"]
    index: int = 0

    for bc in bcs:
        for x_p in xs_p:
            run(N, N_p, x_p, bc, frames,
                index=index,
                path=f"{path}/{bc}")