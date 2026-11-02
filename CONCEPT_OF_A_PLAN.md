# Concept of a Plan for Simulating `The Brick`

This document summarizes the conceptual plan for simulating `The Brick`. It outlines the key components, solvers, and sources of randomness that will be used in the simulation.

----

## 1. Creating the Brick

This I believe is largely done, as it is basically a turbulent spherical molecular cloud that is reshaped into the elongated, curved form of the Brick (G0.253+0.016). It is implemented in `create_cloud()` in `brick/create_cloud.py`, with all parameters in `brick/properties.py`.

### Steps

1. **Spherical turbulent cloud**: AMUSE's `molecular_cloud` (`amuse.ext.molecular_cloud`) generates `N` gas particles of total mass `M` inside a sphere of radius `R`, laid out on a body-centred grid and given a turbulent velocity field.
2. **Elongation** (`shape_brick`): stretch along x by `s = aspect_ratio^(2/3)` and squeeze y and z by `1/sqrt(s)`, so that length / width = `aspect_ratio`. This keeps the volume the same, so the mean density stays the same.
3. **Bending** (`shape_brick`): shift y by `bend * (x / half_length)^2`, a parabola in the x-y plane, so that the ends sit `bend` away from the middle (the sagitta).
4. **Irregularities** (`add_irregularities`): move every particle by a smooth random field, the sum of `n_modes = 20` plane waves with random directions, phases and wavelengths (spread evenly in log between a minimum and a maximum). The rms displacement per coordinate is `IRREGULARITY`. This simulates `unknown` lumpy structure, by creating denser and emptier patches inside. The motivation comes from how we model the universe on large scales with a superposition of waves to create structure (should be familiar if you took the `Large Scale Structure and Galaxy Formation` course by Joop Schaye).

### Current parameters (`properties.py`)

| Parameter | Value | Meaning |
|---|---|---|
| `MASS` | 10e5 M☉ (= 10^6 M☉) | Total cloud mass |
| `RADIUS` | 2.5 pc | Radius of the sphere before shaping (equal-volume radius) |
| `NUMBER_OF_PARTICLES` | 1000 | SPH particles (low, fine for testing) |
| `ASPECT_RATIO` | 3.0 | Length / width |
| `BEND` | 1.0 pc | Sagitta of the curve |
| `IRREGULARITY` | 0.4 pc | rms displacement (0 pc = smooth cloud) |
| `IRREGULARITY_MIN/MAX_WAVELENGTH` | 1.0 / 8.0 pc | Smallest lumps / largest wiggles |
| `IRREGULARITY_SEED` | 42 | Seed of the random displacement field |

### Sources of randomness

- The turbulent velocity field and particle layout of `molecular_cloud` (currently fixed at `seed=100`). This controls the initial randomness of the cloud's structure and velocities.
- The irregularity field (`IRREGULARITY_SEED`, the number of modes, the amplitude and the wavelength range). This controls the small-scale lumpy structure inside the cloud. Since these are apriori unknown to use due to imperfect observational data we need to explore different realizations to get a "mean" representation of the cloud's structure.

The effect of these should be tested by running several seeds and varying the irregularity settings. Five should be sufficient to get a sense of the variability. 

### Diagnostics

The initial cloud can be checked with `plot_cloud` (2D column density), `plot_cloud_3d` (particles plus projections on the walls) and `plot_cloud_volume` (interactive plotly volume rendering).

### Still to do

- Verify the `MASS` against the literature. The Brick is usually quoted at about 10^5 M☉.
- Raise `N` for actual runs.
- Decide whether the turbulent velocity field should also be reshaped (stretched or bent) together with the positions, right now it is not, meaning it follows the original spherical distribution. I believe this is a reasonable approximation for now, but it may need to be revisited for more accurate modeling of the cloud's dynamics.

----

## 2. Run Hydrodynamics Simulation

Once the Brick is created, we evolve it with a hydrodynamics code to see how the gas behaves under its own gravity and pressure. This is implemented in `run_molecular_cloud()` in `brick/create_cloud.py`. It builds the cloud in the same way as `create_cloud()`, hands the particles to the SPH code and evolves them in time.

### Current parameters (`run_molecular_cloud`)

| Parameter | Value | Meaning |
|---|---|---|
| `eps` | 0.1 pc | Smoothing length and gravitational softening (the same for all particles) |
| `dt` | 0.01 Myr | Time step, and the interval between outputs |
| `t_end` | 1.0 Myr | End time of the simulation |
| `gamma` | 1 | Isothermal equation of state |

These are hard-coded in `run_molecular_cloud`, not in `properties.py`.

### Solvers

Right now we are using a simple solver for the hydrodynamics simulation - SPH (Smoothed Particle Hydrodynamics). Currently, it does the following:

- **Code**: AMUSE's `Fi`, a tree-SPH code that computes both the hydrodynamics and the self-gravity of the gas.
- **Thermodynamics**: isothermal gas (`gamma = 1`, `isothermal_flag = True`), with no radiative heating or cooling (`radiation_flag = False`). The temperature stays at whatever the initial internal energy from `molecular_cloud` corresponds to.
- **Resolution**: a fixed smoothing length and gravitational softening of `eps = 0.1 pc` for all particles (`eps_is_h_flag = False`), so the smoothing length does not adapt to the local density.

These were the default settings in the code I used to simulate the molecular cloud. The original version is in https://github.com/amusecode/amuse/blob/main/src/amuse/examples/molecular_cloud_collapse.py, my code just adapts their template and uses it for the specific case of the Brick.

We need to see if we should switch to a more sophisticated solver or if SPH is sufficient for our purposes.

### Sources of randomness

The SPH evolution itself is deterministic, so there are no sources of randomness during the simulation itself.

### Diagnostics

- **Energy conservation**: at every step the total energy (kinetic + potential + thermal) is compared with its value at the start, and the relative error is printed. This tests how accurate the solver is.
- **Density slices**: `plot_hydro` saves a map of the density in the z = 0 plane, sampled directly from the SPH code (`GMC_<i>.png`).
- **Volume renderings**: `plot_cloud_volume` saves a 3D density frame at every step (`GMC_volume_<i>.png`). The frames use a fixed box and colour scale, so they can be put together into a movie with ffmpeg.

Output files get the prefix `GMC_irregular_` instead of `GMC_` when irregularities are switched on.

### Still to do

- Refactor the code to use a separate function for creating the molecular cloud, so that the initial conditions can be easily modified and reused across different simulations. Right now it is very tightly coupled with the creation of the cloud within `run_molecular_cloud()`.

----

## 3. Run Gravity-Only Simulation with a probe particle

We need to setup the initial conditions for the gravity-only simulation. I believe that there is no reliable measurments of the proper velocity of the cloud. Therefore, we may need to assume multiple possible initial conditions, such as different velocity vectors for the cloud and distance from the Galactic center.

### Probe Particle Initial Conditions

Initially we can model the cloud as a single probe particle representing the entire cloud. This simplification can let us play around with different initial conditions without having to deal with the full complexity of the cloud. We can use this to explore the space of possible initial conditons and understand the general behavior of the cloud under gravity-only dynamics.

- `POSITION` | (x, y, z) | Initial position of the probe particle relative to the galactic center |
- `VELOCITY` | (vx, vy, vz) | Initial velocity of the probe particle relative to the galactic center |

### Gravity Potential

The gravitational potential in which the probe particle moves can be modeled as the combined potential of the Galactic center and the surrounding mass distribution. The simplest form of this potential is the point-mass potential of the Galactic center:

$$
\Phi(\mathbf{r}) = -\frac{G M_\mathrm{BH}}{|\mathbf{r}|}
$$

where \(G\) is the gravitational constant, \(M_\mathrm{BH}\) is the mass of Sagittarius A* (the supermassive black hole at the Galactic center), and \(\mathbf{r}\) is the position vector of the probe particle relative to the Galactic center. More complex potentials. I am not sure if including contributions from the Galactic disk and/or halo are needed but I belive they can easily be added later if needed.

- `M_\mathrm{BH}` | Mass of Sagittarius A* $\aprox 4 \times 10^6 M_\odot$ | Determines the strength of the gravitational potential |

This is very simplistic model. To accurately model the gravitational potential we need to combine Nuclear Star Cluster (NSC), Nuclear Stellar Disk (NSD), the Bulge, and Sagittarius A*. Each of these components contributes to the overall gravitational potential and can be modeled separately before combining them into a total potential. However, I think this is too much work, we might be good if we just set the potential to be dominated by Sagittarius A*.

On the other hand there seems to be a module for that in AMUSE. The `amuse.ext.galactic_potentials` module maybe has everything we need to model the combined gravitational potential of the Galactic center, disk, and halo.

### Sources of randomness

In the gravity-only simulation, the main sources of randomness come from the initial conditions of the probe particle. Since we do not have precise measurements of the cloud's proper velocity, we may need to explore a range of possible initial velocities and positions.

### Diagnostics

In the gravity-only simulation, we can use several diagnostics to understand the behavior of the probe particle:

- **Trajectory**: Track the position of the probe particle over time to visualize its orbit around the Galactic center.
- **Velocity**: Monitor the velocity of the probe particle to see how it changes under the influence of the gravitational potential.
- **Energy**: Calculate the kinetic and potential energy of the probe particle to check for conservation of energy. Should be constant in a central gravitational potential.
- **Angular Momentum**: Compute the angular momentum of the probe particle to see if it is conserved. Should be constant in a central gravitational potential.

### Solvers

We need to check this.

----

## 4. Run Star Formation

We need to figure out what to do with star formation if we wish to simulate it at all. There is a module with sinks (`amuse.ext.sink`) that can be useful.

### Sources of randomness

...

### Diagnostics

...

### Solvers

...

----

## 5. Unifying Everything

After all individual components of the simulation have been developed and tested separately, we need to integrate them into a unified framework. This involves ensuring that the gravity-only simulation, star formation processes, and any other physical processes interact consistently and correctly within the overall simulation environment. This should be straightforward if each component has well-defined interfaces and adheres to the same units and conventions.

### Useful sources from AMUSE

**Example scripts (`examples/`)**

Browse them at https://github.com/amusecode/amuse/tree/main/src/amuse/examples.

| Topic | Examples |
|---|---|
| Molecular clouds | `molecular_cloud_collapse.py` (the template for our code), `molecular_cloud_collapse_with_sinks.py` + `plot_molecular_cloud_collapse_with_sinks.py`, `molecular_cloud_chemistry.py`, `cloudshock.py`, `boss_bodenheimer_test.py` |
| Orbits in an external potential | `evolve_orbit_in_potential.py`, `gravity_potential.py`, `Sun_and_M67_in_the_Galaxy.py` |
| Cluster in a galaxy | `solar_cluster_in_galaxy_potential.py`, `solar_cluster_in_semilive_galaxy.py`, `solar_cluster_in_live_galaxy.py` |
| Coupling codes with Bridge | `gravity_hydro.py`, `gas_in_cluster.py`, `particles_and_gas_in_cluster.py`, `three_body_bridge.py`, `three_body_bridge_hierarchical.py` |
| Sinks and star formation | `hydro_sink_particles.py`, `test_jeans_instability.py`, `salpeter.py` |
| Minimal templates | `hydro_minimal.py`, `gravity_minimal.py`, `stellar_minimal.py` |

Note: Found using Claude so do double-check the examples and references for accuracy.

### Coupling everything

- `amuse.couple.bridge.Bridge`: the standard way to couple codes in AMUSE. It combines a hydro code, a gravity code and/or an external potential. This is how the Brick (Fi) can be put into the Galactic-centre potential.
- `nbody_system.nbody_to_si`: unit converter. All codes in a bridge need consistent converters.
