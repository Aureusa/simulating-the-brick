"""
TheBrick: a wrapper around the gas particles that represent the Brick
(G0.253+0.016).

The class only *represents* the cloud: it builds it, holds its particles and
parameters, gives physical diagnostics, places it in a (Galactic) frame and
connects it to AMUSE codes. The solvers themselves (hydro, gravity, bridge,
star formation) are set up elsewhere and act on a TheBrick instance through
`add_to`, `sync_from` and `sync_to`.

Typical use:

    brick = TheBrick.from_properties()
    brick.place_at(position, velocity)       # in the Galactic-centre frame
    sph = Fi(brick.converter)
    brick.add_to(sph)
    ...
    sph.evolve_model(t)
    brick.sync_from(sph)
"""
import numpy

from amuse.units import units, constants, nbody_system
from amuse.datamodel import Particles
from amuse.io import write_set_to_file, read_set_from_file
from amuse.ext.molecular_cloud import molecular_cloud
from amuse.ext.evrard_test import body_centered_grid_unit_cube

import properties as prop
from create_cloud import (shape_brick, add_irregularities, density_grid,
                          plot_cloud, plot_cloud_3d, plot_cloud_volume)


class TheBrick:
    """
    The Brick as a set of SPH gas particles plus the parameters it was made
    with. `self.gas` is the AMUSE particle set; everything else works on it.
    """

    def __init__(self, gas, converter, Mcloud, Rcloud, aspect_ratio=1.,
                 bend=0. | units.parsec, irregularity=0. | units.parsec,
                 half_length=None, cloud_seed=None, irregularity_seed=None):
        self.gas = gas
        self.converter = converter
        self.Mcloud = Mcloud
        self.Rcloud = Rcloud
        self.aspect_ratio = aspect_ratio
        self.bend = bend
        self.irregularity = irregularity
        self.half_length = (half_length if half_length is not None
                            else aspect_ratio**(2./3.)*Rcloud)
        self.cloud_seed = cloud_seed
        self.irregularity_seed = irregularity_seed

        # Model time of the particles, updated by sync_from
        self.time = 0. | units.Myr
        # One (from_code, to_code) channel pair per attached code
        self._channels = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def create(cls, N=100, Mcloud=100. | units.MSun,
               Rcloud=1. | units.parsec, aspect_ratio=1.,
               bend=0. | units.parsec, irregularity=0. | units.parsec,
               cloud_seed=100, irregularity_seed=prop.IRREGULARITY_SEED,
               min_wavelength=prop.IRREGULARITY_MIN_WAVELENGTH,
               max_wavelength=prop.IRREGULARITY_MAX_WAVELENGTH,
               n_modes=20):
        """
        Build the Brick like `create_cloud` does: a turbulent spherical cloud
        from `molecular_cloud`, stretched and bent by `shape_brick`, and
        (if irregularity > 0) perturbed by `add_irregularities`.
        Unlike `create_cloud`, both random seeds are parameters.
        """
        converter = nbody_system.nbody_to_si(Mcloud, Rcloud)
        gas = molecular_cloud(targetN=N, convert_nbody=converter,
                              base_grid=body_centered_grid_unit_cube,
                              seed=cloud_seed).result
        half_length = shape_brick(gas, Rcloud, aspect_ratio, bend)

        if irregularity > 0 | units.parsec:
            add_irregularities(gas, irregularity, min_wavelength,
                               max_wavelength, n_modes=n_modes,
                               seed=irregularity_seed)

        return cls(gas, converter, Mcloud, Rcloud, aspect_ratio, bend,
                   irregularity, half_length, cloud_seed, irregularity_seed)

    @classmethod
    def from_properties(cls, **overrides):
        """
        Build the Brick from the values in properties.py. Any keyword of
        `create` can be overridden, e.g. from_properties(irregularity_seed=7).
        """
        kwargs = dict(N=int(prop.NUMBER_OF_PARTICLES.number),
                      Mcloud=prop.MASS,
                      Rcloud=prop.RADIUS,
                      aspect_ratio=prop.ASPECT_RATIO,
                      bend=prop.BEND,
                      irregularity=prop.IRREGULARITY)
        kwargs.update(overrides)
        return cls.create(**kwargs)

    @classmethod
    def realisations(cls, seeds, **kwargs):
        """
        One Brick per irregularity seed, all other parameters equal.
        Used to sample the unknown internal structure (see the plan).
        """
        return [cls.from_properties(irregularity_seed=s, **kwargs)
                for s in seeds]

    def copy(self):
        """Independent copy (particles copied, channels not)."""
        new = TheBrick(self.gas.copy(), self.converter, self.Mcloud,
                       self.Rcloud, self.aspect_ratio, self.bend,
                       self.irregularity, self.half_length, self.cloud_seed,
                       self.irregularity_seed)
        new.time = self.time
        return new

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.gas)

    def __repr__(self):
        return ("TheBrick(N={0}, M={1}, R={2}, aspect_ratio={3}, bend={4}, "
                "irregularity={5}, t={6})").format(
                    len(self), self.mass.in_(units.MSun),
                    self.Rcloud.in_(units.parsec), self.aspect_ratio,
                    self.bend.in_(units.parsec),
                    self.irregularity.in_(units.parsec),
                    self.time.in_(units.Myr))

    @property
    def mass(self):
        return self.gas.mass.sum()

    @property
    def position(self):
        """Centre of mass position."""
        return self.gas.center_of_mass()

    @property
    def velocity(self):
        """Centre of mass velocity."""
        return self.gas.center_of_mass_velocity()

    @property
    def mean_density(self):
        """Mean density of the cloud before shaping (shaping keeps volume)."""
        return 3.*self.Mcloud/(4.*numpy.pi*self.Rcloud**3)

    @property
    def free_fall_time(self):
        """t_ff = sqrt(3 pi / (32 G rho)), as in create_cloud.py."""
        return 0.5427/(constants.G*self.mean_density).sqrt()

    # ------------------------------------------------------------------
    # Energies and dynamical state
    # ------------------------------------------------------------------
    def kinetic_energy(self):
        """Kinetic energy in the centre-of-mass frame."""
        v = self.gas.velocity - self.velocity
        return 0.5*(self.gas.mass*(v**2).sum(axis=1)).sum()

    def potential_energy(self):
        """Self-gravitational energy. O(N^2): slow for large N."""
        return self.gas.potential_energy(G=constants.G)

    def thermal_energy(self):
        return self.gas.thermal_energy()

    def total_energy(self):
        return (self.kinetic_energy() + self.potential_energy()
                + self.thermal_energy())

    def virial_ratio(self):
        """Q = (E_kin + E_th) / |E_pot|. Q < 0.5 means the cloud collapses."""
        return ((self.kinetic_energy() + self.thermal_energy())
                / abs(self.potential_energy()))

    def velocity_dispersion(self):
        """3D rms velocity in the centre-of-mass frame."""
        v = self.gas.velocity - self.velocity
        return ((v**2).sum(axis=1).mean()).sqrt()

    def lagrangian_radius(self, fraction=0.5):
        """Radius around the centre of mass containing `fraction` of the mass."""
        r = (self.gas.position - self.position).lengths()
        order = numpy.argsort(r.value_in(units.parsec))
        cum = numpy.cumsum(self.gas.mass[order].value_in(units.MSun))
        i = numpy.searchsorted(cum, fraction*cum[-1])
        return r[order][min(i, len(r)-1)]

    def jacobi_radius(self, enclosed_mass, distance=None):
        """
        Tidal (Jacobi) radius for a point-mass host: r_J = d (m / 3M)^(1/3).
        `distance` defaults to |position|, i.e. distance to the origin of the
        frame (the Galactic centre after place_at).
        """
        if distance is None:
            distance = self.position.length()
        return distance*(self.mass/(3.*enclosed_mass))**(1./3.)

    # ------------------------------------------------------------------
    # Placement in a (Galactic) frame
    # ------------------------------------------------------------------
    def move_to_center(self):
        """Put the centre of mass at rest at the origin."""
        self.gas.move_to_center()
        return self

    def place_at(self, position, velocity=None):
        """
        Move the Brick so that its centre of mass is at `position` with bulk
        velocity `velocity` (e.g. in the Galactic-centre frame).
        The internal structure and velocities are kept.
        """
        if velocity is None:
            velocity = [0., 0., 0.] | units.kms
        self.gas.position += position - self.position
        self.gas.velocity += velocity - self.velocity
        return self

    def rotate(self, angle, axis=(0., 0., 1.)):
        """
        Rotate positions and velocities by `angle` (radians, or an AMUSE angle
        quantity) around `axis` through the centre of mass. Used to orient
        the long axis of the Brick (x) in the Galactic frame.
        """
        if hasattr(angle, "value_in"):
            angle = angle.value_in(units.rad)
        k = numpy.asarray(axis, dtype=float)
        k /= numpy.linalg.norm(k)
        K = numpy.array([[0., -k[2], k[1]],
                         [k[2], 0., -k[0]],
                         [-k[1], k[0], 0.]])
        # Rodrigues' rotation formula
        Rm = (numpy.eye(3) + numpy.sin(angle)*K
              + (1.-numpy.cos(angle))*K @ K)

        com, vcom = self.position, self.velocity
        pos = (self.gas.position - com).value_in(units.parsec)
        vel = (self.gas.velocity - vcom).value_in(units.kms)
        self.gas.position = (pos @ Rm.T | units.parsec) + com
        self.gas.velocity = (vel @ Rm.T | units.kms) + vcom
        return self

    def as_probe_particle(self):
        """
        The whole Brick as a single particle (total mass, centre-of-mass
        position and velocity), for the gravity-only runs of Section 3.
        Returned as a Particles set of length 1 so it can be added to a code.
        """
        probe = Particles(1)
        probe.mass = self.mass
        probe.position = self.position
        probe.velocity = self.velocity
        probe.radius = self.half_length
        probe.name = "TheBrick"
        return probe

    def set_smoothing_length(self, h):
        """Fixed SPH smoothing length for all particles (as in run_molecular_cloud)."""
        self.gas.h_smooth = h
        return self

    # ------------------------------------------------------------------
    # Coupling to AMUSE codes
    # ------------------------------------------------------------------
    def add_to(self, code, particle_set="gas_particles"):
        """
        Add the gas to `code` (e.g. code.gas_particles of Fi, or
        code.particles of a gravity code) and open channels both ways.
        Returns the code's particle set.
        """
        target = getattr(code, particle_set)
        target.add_particles(self.gas)
        self._channels[code] = (target.new_channel_to(self.gas),
                                self.gas.new_channel_to(target))
        return target

    def sync_from(self, code, attributes=None):
        """Copy the state of `code` back into self.gas and update self.time."""
        channel = self._channels[code][0]
        if attributes is None:
            channel.copy()
        else:
            channel.copy_attributes(attributes)
        if hasattr(code, "model_time"):
            self.time = code.model_time
        return self

    def sync_to(self, code, attributes=None):
        """Copy self.gas into `code` (e.g. after changing particles here)."""
        channel = self._channels[code][1]
        if attributes is None:
            channel.copy()
        else:
            channel.copy_attributes(attributes)
        return self

    def detach(self, code):
        """Forget the channels to `code` (call before code.stop())."""
        self._channels.pop(code, None)
        return self

    # ------------------------------------------------------------------
    # Diagnostics and I/O
    # ------------------------------------------------------------------
    def density_grid(self, L=None, N=64, smoothing=0.2):
        """Smoothed 3D density grid; see create_cloud.density_grid."""
        return density_grid(self.gas, L, N, smoothing)

    def plot_column_density(self, **kwargs):
        plot_cloud(self.gas, **kwargs)

    def plot_3d(self, **kwargs):
        plot_cloud_3d(self.gas, **kwargs)

    def plot_volume(self, **kwargs):
        plot_cloud_volume(self.gas, **kwargs)

    def summary(self):
        """Dictionary of the main diagnostics (handy for logging per snapshot)."""
        return dict(time=self.time.in_(units.Myr),
                    mass=self.mass.in_(units.MSun),
                    position=self.position.in_(units.parsec),
                    velocity=self.velocity.in_(units.kms),
                    half_mass_radius=self.lagrangian_radius(0.5).in_(units.parsec),
                    velocity_dispersion=self.velocity_dispersion().in_(units.kms),
                    virial_ratio=self.virial_ratio())

    def save(self, filename):
        """Write the particles and the parameters to an AMUSE (HDF5) file."""
        attrs = self.gas.collection_attributes
        attrs.Mcloud = self.Mcloud
        attrs.Rcloud = self.Rcloud
        attrs.aspect_ratio = self.aspect_ratio
        attrs.bend = self.bend
        attrs.irregularity = self.irregularity
        attrs.half_length = self.half_length
        attrs.time = self.time
        write_set_to_file(self.gas, filename, "amuse", overwrite_file=True)

    @classmethod
    def load(cls, filename):
        """Read a Brick written by `save`."""
        gas = read_set_from_file(filename, "amuse").copy()
        attrs = gas.collection_attributes

        def get(name, default):
            try:
                value = getattr(attrs, name)
                return default if value is None else value
            except AttributeError:
                return default

        Mcloud = get("Mcloud", gas.mass.sum())
        Rcloud = get("Rcloud", prop.RADIUS)
        brick = cls(gas, nbody_system.nbody_to_si(Mcloud, Rcloud), Mcloud,
                    Rcloud, get("aspect_ratio", 1.),
                    get("bend", 0. | units.parsec),
                    get("irregularity", 0. | units.parsec),
                    get("half_length", None))
        brick.time = get("time", 0. | units.Myr)
        return brick


if __name__ == "__main__":
    brick = TheBrick.from_properties()
    print(brick)
    print("t_ff =", brick.free_fall_time.in_(units.Myr))
    print(brick.summary())
