#!/usr/bin/env python3

from itertools import combinations
import logging
import os
import platform
import sys

import numpy as np
import openmm
from openmm import app, unit
from scipy import optimize

from mdcraft.openmm.pair import coul_gauss, gauss
from mdcraft.openmm.reporter import NetCDFReporter
from mdcraft.openmm.system import (register_particles, add_image_charges,
                                   add_slab_correction)
from mdcraft.openmm.topology import create_atoms
from mdcraft.openmm.unit import get_lj_scale_factors

KAPPA_INV = 15.9835
OMEGA = 0.499

def run(N: int, bc: str, frames: int, *, dt_md: float = 0.02,
        sigma_q: unit.Quantity = 0 * unit.elementary_charge / unit.nanometer ** 2,
        temperature: unit.Quantity = 300 * unit.kelvin,
        size: unit.Quantity = 0.275 * unit.nanometer,
        mass: unit.Quantity = 18.01528 * unit.gram / unit.mole,
        N_m: float = 4.0, rho_reduced: float = 2.5, u_shift_reduced: float = 1e-3,
        varepsilon_r: float = 78.0, a_scale: float = 1.0, every: int = 1_000,
        L_z_scale: float = 2.5, device: int = 0, index: int = None,
        path: str = os.getcwd(), verbose: bool = True) -> None:

    logging.basicConfig(format="{asctime} | {levelname:^8s} | {message}",
                        style="{",
                        level=logging.INFO if verbose else logging.WARNING)

    if not os.path.isdir(path):
        os.makedirs(path)
        logging.info(f"Created data directory '{path}'.")
    os.chdir(path)
    logging.info(f"Changed to data directory '{path}'.")

    scales = get_lj_scale_factors(
        {
            "energy": (unit.BOLTZMANN_CONSTANT_kB * temperature)
                      .in_units_of(unit.kilojoule),
            "length": size * (N_m * rho_reduced) ** (1 / 3) if N_m > 1 else size,
            "mass": mass * N_m
        }
    )
    logging.info("Computed scaling factors for reducing physical quantities.\n"
                 "  Fundamental quantities:\n"
                 f"    Molar energy: {scales['molar_energy']}\n"
                 f"    Length: {scales['length']}\n"
                 f"    Mass: {scales['mass']}")

    rho = rho_reduced / scales["length"] ** 3
    box_length_nd = ((N / (L_z_scale * rho)) ** (1 / 3)).value_in_unit(unit.nanometer)
    positions_wall, dimensions = create_atoms(
        np.array((box_length_nd, box_length_nd, 0)),
        lattice="hcp",
        length=scales["length"] / 2,
        flexible=True
    )
    dimensions[2] = N / (rho * dimensions[0] * dimensions[1])

    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        *(dimensions * np.diag(np.ones(3, dtype=float)))
    )
    topology = app.Topology()
    topology.setUnitCellDimensions(dimensions)
    logging.info("Created simulation system and topology with "
                 f"dimensions {dimensions[0]} x {dimensions[1]} "
                 f"x {dimensions[2]}.")

    # Types: real particle (p), wall (w), image charge (i)
    N_gauss_types = 3 if bc == "ic" else 2 if bc == "slab" else 1
    radius_nd = scales["length"].value_in_unit(unit.nanometer) / 2
    sigmas_i_sq = (np.array((radius_nd, 0, radius_nd)) * unit.nanometer) ** 2
    sigmas_ij_sq = sigmas_i_sq + sigmas_i_sq[:, None]
    betas_ij = 3 / (2 * sigmas_ij_sq)
    alphas_ij_coefs = 1 + np.array((
        (0, 0, -1),     # pp, pw, pi;
        (0, -1, -1),    # wp, ww, wi;
        (-1, -1, -1)    # ip, iw, ii
    ))
    A_md = (N_m * KAPPA_INV - 1) / (2 * OMEGA * rho_reduced)
    A = A_md * scales["molar_energy"] * scales["length"] ** 3
    alphas_ij = alphas_ij_coefs * A * (betas_ij / np.pi) ** (3 / 2)
    alphas_ij[np.isnan(alphas_ij)] = 0 * unit.kilojoule_per_mole
    cutoff = optimize.fsolve(
        lambda r: np.max(alphas_ij)
                  * np.exp(-np.min(betas_ij) * (r * unit.nanometer) ** 2)
                  / scales["molar_energy"] - u_shift_reduced,
        scales["length"].value_in_unit(unit.nanometer)
    )[0] * unit.nanometer
    pair_gauss = gauss(
        cutoff,
        mix="alpha12=alpha(type1,type2);beta12=beta(type1,type2);",
        per_params=("type",),
        tab_funcs={"alpha": alphas_ij[:N_gauss_types, :N_gauss_types],
                   "beta": betas_ij[:N_gauss_types, :N_gauss_types]}
    )

    # Types: real or image particle (p), wall (w)
    N_elec_types = 1 if bc == "bulk" else 2
    as_i_sq = (np.array((a_scale, 0)) * scales["length"] / 2) ** 2
    as_ij = (as_i_sq + as_i_sq[:, None]) ** (1 / 2) # pp, pw; wp, ww
    e = 1 * unit.elementary_charge
    q_scaled = e / np.sqrt(varepsilon_r)
    pair_elec_dir, pair_elec_rec = coul_gauss(
        cutoff,
        mix="alpha12=alpha(type1,type2);",
        per_params=("type",),
        tab_funcs={"alpha": np.sqrt(np.pi / 2)
                            / as_ij[:N_elec_types, :N_elec_types]}
    )

    system.addForce(pair_gauss)
    system.addForce(pair_elec_dir)
    system.addForce(pair_elec_rec)
    logging.info(f"Registered {system.getNumForces()} pair "
                 "potential(s) to the simulation.")

    element_anion = app.Element.getBySymbol("Cl")
    element_cation = app.Element.getBySymbol("Na")
    element_wall = app.Element.getBySymbol("C")

    N_a = N_c = N // 2

    register_particles(
        system, topology, N_c, scales["mass"],
        element=element_cation,
        name="CAT",
        nbforce=pair_elec_rec,
        charge=q_scaled,
        cnbforces={pair_elec_dir: (q_scaled, 0), pair_gauss: (0,)}
    )
    logging.info(f"Registered {N_c:,} cation(s) to the simulation.")

    register_particles(
        system, topology, N_a, scales["mass"],
        element=element_anion,
        name="ANI",
        nbforce=pair_elec_rec,
        charge=-q_scaled,
        cnbforces={pair_elec_dir: (-q_scaled, 0), pair_gauss: (0,)}
    )
    logging.info(f"Registered {N_a:,} anion(s) to the simulation.")

    if bc != "bulk":
        positions_wall = np.concatenate((
            positions_wall,
            positions_wall + np.array(
                (0, 0, dimensions[2].value_in_unit(unit.nanometer))
            ) * unit.nanometer
        ))
        N_wall = positions_wall.shape[0]
        q_w_scaled = ((2 if bc == "ic" else 1) * sigma_q
                      * dimensions[0] * dimensions[1]
                      / (N_wall * np.sqrt(varepsilon_r)))

        for sign, name in zip((1, -1), ("LWL", "RWL")):
            register_particles(
                system, topology, N_wall // 2, 0,
                element=element_wall,
                name=name,
                nbforce=pair_elec_rec,
                charge=sign * q_w_scaled,
                cnbforces={pair_elec_dir: (sign * q_w_scaled, 1),
                           pair_gauss: (1,)}
            )
        logging.info(f"Registered {N_wall:,} wall particles to the force field.")

        wall_indices = range(N, N + N_wall)
        for i, j in combinations(wall_indices, 2):
            pair_elec_dir.addExclusion(i, j)
            pair_elec_rec.addException(i, j, 0, 0, 0)
            pair_gauss.addExclusion(i, j)
        logging.info("Removed wall–wall interactions.")

    if index is None:
        index = 0
    if bc == "bulk":
        filename = f"bulk_dtmd_{dt_md}_openmm"
    else:
        sigma_q_nd = sigma_q.value_in_unit(unit.elementary_charge
                                           / unit.nanometer ** 2)
        filename = f"{bc}_sigmaq_{sigma_q_nd}_dtmd_{dt_md}_openmm"

    platform_ = openmm.Platform.getPlatformByName("CUDA")
    properties = {"Precision": "mixed", "DeviceIndex": str(device),
                  "UseBlockingSync": "false"}
    dt = dt_md * scales["time"]
    friction = 1e-3 / dt
    logging.info(f"Initialized the {platform_.getName()} platform in OpenMM "
                 f"{platform_.getOpenMMVersion()} on {platform.node()}.")

    while True:
        positions = create_atoms(dimensions, N)
        if bc != "bulk":
            z_factor = 0.05
            z_wall = 2 ** (-5 / 6) * scales["length"]
            z_scale = (((1 - z_factor) * dimensions[2] - z_wall)
                       / (np.max(positions[:, 2]) - np.min(positions[:, 2])))
            positions *= np.array((1, 1, z_scale))
            positions[:, 2] += ((z_wall + z_factor * dimensions[2]) / 2
                                - z_scale * np.min(positions[:, 2]))
            positions = np.concatenate((positions, positions_wall)) * unit.nanometer
        logging.info("Generated random initial configuration for "
                     f"{N:,} particles.")

        logging.info("Starting system relaxation...")
        integrator = openmm.LangevinMiddleIntegrator(temperature, friction, dt)
        simulation = app.Simulation(topology, system, integrator, platform_,
                                    properties)
        simulation.context.setPositions(positions)
        simulation.minimizeEnergy()
        positions = (simulation.context.getState(getPositions=True)
                     .getPositions(asNumpy=True))
        if bc == "bulk" or (positions[:N, 2].min() > 0 * unit.nanometer
                            and positions[:N, 2].max() < dimensions[2]):
            simulation.context.setVelocitiesToTemperature(temperature)
            logging.info("Local energy minimization completed.")
            break
        logging.warning("Particles have escaped the simulation box! "
                        "Trying again...")

    if bc == "ic":
        positions, integrator = add_image_charges(
            system, topology, positions, temperature, friction, dt,
            wall_indices=wall_indices,
            nbforce=pair_elec_rec,
            cnbforces={pair_elec_dir: {"charge": 0},
                       pair_gauss: {"replace": {0: 2}}}
        )
    elif bc == "slab":
        integrator = add_slab_correction(system, topology, pair_elec_rec,
                                         temperature, friction, dt)

    if bc != "bulk":
        simulation = app.Simulation(topology, system, integrator, platform_,
                                    properties)
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(temperature)

    with open(f"{filename}.cif", "w") as f:
        app.PDBxFile.writeFile(simulation.topology, positions, f, keepIds=True)
    logging.info(f"Wrote topology to '{filename}.cif'.")

    simulation.reporters.append(
        app.CheckpointReporter(f"{filename}.chk", 100 * every)
    )
    logging.info("Registered checkpoint reporter writing to "
                 f"'{filename}.cif' to the simulation.")
    simulation.reporters.append(NetCDFReporter(f"{filename}.nc", every))
    logging.info("Registered trajectory reporter writing to "
                 f"'{filename}.nc' to the simulation.")
    timesteps = frames * every
    for o in [sys.stdout, f"{filename}.log"]:
        simulation.reporters.append(
            app.StateDataReporter(
                o, reportInterval=every, step=True, temperature=True,
                volume=True, potentialEnergy=True, kineticEnergy=True,
                totalEnergy=True, remainingTime=True, speed=True,
                totalSteps=timesteps
            )
        )
    logging.info("Registered state data reporter writing to "
                 f"'{filename}.log' to the simulation.")

    logging.info(f"Starting NVT run with {timesteps:,} timesteps...")
    simulation.step(timesteps)
    simulation.saveState(f"{filename}.xml")
    logging.info("Simulation completed. Wrote final simulation state "
                 f"to '{filename}.xml'.")

if __name__ == "__main__":

    path: str = "/mnt/e/research/gcme/performance/gcme"
    N: int = 1_000
    frames: int = 1_000

    bcs: list[str] = ["bulk", "ic", "slab"]
    dts_md: list[float] = [0.005, 0.02, 0.05]

    for bc in bcs:
        for dt_md in dts_md:
            run(N, bc, frames,
                dt_md=dt_md,
                sigma_q=0.005 * unit.elementary_charge / unit.nanometer ** 2,
                path=path)