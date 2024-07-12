#!/usr/bin/env python3

from itertools import combinations
import logging
import os
import platform
import sys

import numpy as np
import openmm
from openmm import app, unit

from mdcraft.openmm.pair import lj_coul, wca
from mdcraft.openmm.reporter import NetCDFReporter
from mdcraft.openmm.system import register_particles, add_image_charges
from mdcraft.openmm.topology import create_atoms
from mdcraft.openmm.unit import get_lj_scale_factors

logging.basicConfig(format="{asctime} | {levelname:^8s} | {message}",
                    style="{", level=logging.INFO)

path = "/mnt/e/research/gcme/data/performance/ljcoul"
os.chdir(path)
logging.info(f"Changed to data directory '{path}'.")

N = 1_000
rho_reduced = 0.8
temperature = 300 * unit.kelvin
mass = 72 * unit.gram / unit.mole
sigma = 0.3 * unit.nanometer
L_z_scale = 2.5
e = 1 * unit.elementary_charge
varepsilon_r = 78.0
sigma_q_reduced = 0.005
sigma_q = sigma_q_reduced * unit.elementary_charge / unit.nanometer ** 2
dt_reduced = 0.005
every = 1_000
timesteps = 100 * every

scales = get_lj_scale_factors(
    {
        "energy": (unit.BOLTZMANN_CONSTANT_kB * temperature)
                  .in_units_of(unit.kilojoule),
        "length": sigma,
        "mass": mass
    }
)

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

# cutoff = 2 ** (1 / 6) * sigma
# pair_wca = wca(cutoff)

# Types: real particle (p), wall (w), image charge (i)
cutoff = 3 * sigma
sigma_nd = sigma.value_in_unit(unit.nanometer)
sigmas = np.array((sigma_nd, 0, sigma_nd)) * unit.nanometer
sigmas_ij = (sigmas + sigmas[:, None]) / 2
epsilon_nd = scales["molar_energy"].value_in_unit(unit.kilojoule / unit.mole)
epsilons = np.array((epsilon_nd, 10_000 * epsilon_nd, 0))
epsilons_ij = np.sqrt(epsilons * epsilons[:, None]) * unit.kilojoule / unit.mole
epsilons_ij[1, 1] = 0 * unit.kilojoule / unit.mole
pair_wca = wca(
    cutoff,
    mix="sigma12=sigma(type1,type2);epsilon12=epsilon(type1,type2);",
    per_params=("type",),
    tab_funcs={"sigma": sigmas_ij, "epsilon": epsilons_ij}
)

pair_coul = lj_coul(cutoff)

system.addForce(pair_wca)
system.addForce(pair_coul)
logging.info(f"Registered {system.getNumForces()} pair "
             "potential(s) to the simulation.")

element_anion = app.Element.getBySymbol("Cl")
element_cation = app.Element.getBySymbol("Na")
element_wall = app.Element.getBySymbol("C")

N_ion = N // 2
q_scaled = e / np.sqrt(varepsilon_r)
for name, element, sign in zip(("cation", "anion"),
                               (element_cation, element_anion), (1, -1)):
    register_particles(
        system, topology, N_ion, scales["mass"], element=element,
        name=name[:3].upper(), resname=name[:3].upper(), nbforce=pair_coul,
        charge=sign * q_scaled,
        # cnbforces={pair_wca: (scales["length"], scales["molar_energy"])}
        cnbforces={pair_wca: (0,)}
    )
    logging.info(f"Registered {N_ion:,} {name}(s) to the force field.")

N_wall = positions_wall.shape[0]
q_wall_scaled = (sigma_q * dimensions[0] * dimensions[1]
                 / (N_wall * np.sqrt(varepsilon_r)))
positions_wall = np.concatenate((
    positions_wall,
    positions_wall + np.array(
        (0, 0, dimensions[2].value_in_unit(unit.nanometer))
    ) * unit.nanometer
))

for sign, name in zip((1, -1), ("LWL", "RWL")):
    register_particles(
        system, topology, N_wall, 0, element=element_wall, name=name,
        nbforce=pair_coul, charge=sign * q_wall_scaled,
        # cnbforces={pair_wca: (0 * unit.nanometer,
        #                       10_000 * scales["molar_energy"])}
        cnbforces={pair_wca: (1,)}
    )
logging.info(f"Registered {N_wall:,} wall particles to the simulation.")

wall_indices = range(N, N + 2 * N_wall)
for i, j in combinations(wall_indices, 2):
    pair_wca.addExclusion(i, j)
    pair_coul.addException(i, j, 0, 0, 0)
logging.info("Removed wall–wall interactions.")

dimensions_electrolyte = dimensions.copy() * unit.nanometer
dimensions_electrolyte[2] -= 2 * scales["length"]
positions = create_atoms(dimensions_electrolyte, N)
positions[:, 2] += scales["length"]
positions = np.concatenate((positions, positions_wall)) * unit.nanometer
logging.info("Generated random initial configuration for "
             f"{N:,} particles and {2 * N_wall:,} wall particles.")

plat = openmm.Platform.getPlatformByName("CUDA")
properties = {"Precision": "mixed", "DeviceIndex": "0",
              "UseBlockingSync": "false"}
dt = dt_reduced * scales["time"]
friction = 1e-2 / dt
logging.info(f"Initialized the {plat.getName()} platform in OpenMM "
             f"{plat.getOpenMMVersion()} on {platform.node()}.")

logging.info("Starting system relaxation...")
integrator = openmm.LangevinMiddleIntegrator(temperature, friction, dt)
simulation = app.Simulation(topology, system, integrator, plat,
                            properties)
simulation.context.setPositions(positions)
simulation.minimizeEnergy()
positions = (simulation.context.getState(getPositions=True)
             .getPositions(asNumpy=True))
if positions[:N, 2].min() > 0 * unit.nanometer \
        and positions[:N, 2].max() < dimensions[2]:
    logging.info("Local energy minimization completed.")
else:
    raise RuntimeError("Local energy minimization failed.")

positions, integrator = add_image_charges(
    system, topology, positions, temperature, friction, dt,
    wall_indices=wall_indices,
    nbforce=pair_coul,
    # cnbforces={pair_wca: None}
    cnbforces={pair_wca: (2,)}
)

simulation = app.Simulation(topology, system, integrator, plat, properties)
simulation.context.setPositions(positions)
simulation.context.setVelocitiesToTemperature(temperature)

filename = f"ic_sigmaq_{sigma_q_reduced}_openmm"
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