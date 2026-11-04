import numpy
from amuse.lab import *
from brick.brick import TheBrick

from .sph import setup_sph, EPS
from .utils import plot_hydro


AVAILABLE_CODES = {
    "Fi": setup_sph
}


def run_hydro(
        brick: TheBrick,
        code: str = "Fi",
        dt: units.Myr = 1.e-2 | units.Myr,
        t_end: units.Myr = 1.0 | units.Myr,
        plot_prefix: str = "hydro"
    ):
    if code not in AVAILABLE_CODES:
        raise ValueError(f"Code '{code}' is not available. Choose from {AVAILABLE_CODES.keys()}.")

    code_name = code
    code_setup_function = AVAILABLE_CODES[code]
    code = code_setup_function(brick.converter, dt)

    # Add the brick's gas particles to the code code (ugly workaround)
    if code_name == "Fi":
        brick.set_smoothing_length(EPS) # Set the code smoothing length for the brick's gas particles

    # Add to code
    brick.add_to(code, particle_set="gas_particles")

    # Get the half-length and mean density of the brick
    half_length = brick.half_length
    rho_cloud = brick.mean_density

    # Things needed for plotting + sims
    i = 0
    L = 2.4*half_length.value_in(units.parsec)  # map covers the whole cloud
    log_rho = numpy.log10(rho_cloud.value_in(units.amu/units.cm**3))
    vmin, vmax = log_rho-2, log_rho+1
    E0 = 0.0
    ttarget = 0.0 | units.Myr

    plot_hydro(ttarget, code, i, L, vmin, vmax, plot_prefix)

    while ttarget < t_end:
        # Advance the target time by one timestep
        ttarget = float(i)*dt
        print("Advancing to t =", ttarget)
        code.evolve_model(ttarget, timestep=dt)
        brick.sync_from(code)

        # Get the total and thermal energy of the brick
        E = brick.total_energy()
        E_th = brick.thermal_energy()

        # The error in total energy compared to the initial value
        # This should stay small if the simulation conserves energy well
        if i == 0:
            E0 = E
        Eerr = (E-E0)/E0
        print('Etot=', E, 'Etot_error=', Eerr, 'E_th=', E_th)

        # Plotting
        plot_hydro(ttarget, code, i, L, vmin, vmax, plot_prefix)
        brick.plot_volume(L=L, vmin=vmin, vmax=vmax, i=i, prefix=plot_prefix)
        i += 1

    # Detach and stop
    brick.detach(code)
    code.stop()
    return brick
