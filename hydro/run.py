import numpy
  
from matplotlib import pyplot 

from amuse.lab import *
from amuse.ext.molecular_cloud import molecular_cloud
from amuse.ext.evrard_test import body_centered_grid_unit_cube

import properties as prop


def run_molecular_cloud(N=100, Mcloud=100. | units.MSun,
                        Rcloud=1. | units.parsec,
                        aspect_ratio=1., bend=0. | units.parsec,
                        irregularity=0. | units.parsec):

    conv = nbody_system.nbody_to_si(Mcloud,Rcloud)

    rho_cloud = 3.*Mcloud/(4.*numpy.pi*Rcloud**3)
    print(rho_cloud)
    tff = 0.5427/numpy.sqrt(constants.G*rho_cloud)
    print("t_ff=", tff.value_in(units.Myr), 'Myr')

    dt = 1.e-2 | units.Myr
    t_end = 1.0 | units.Myr

    parts = molecular_cloud(targetN=N,convert_nbody=conv,
                            base_grid=body_centered_grid_unit_cube,
                            seed=prop.MOLECULAR_CLOUD_SEED).result
    half_length = shape_brick(parts, Rcloud, aspect_ratio, bend)
    prefix = "GMC_"
    if irregularity > 0 | units.parsec:
        add_irregularities(parts, irregularity,
                           prop.IRREGULARITY_MIN_WAVELENGTH,
                           prop.IRREGULARITY_MAX_WAVELENGTH,
                           seed=prop.IRREGULARITY_SEED)
        prefix = "GMC_irregular_"

    sph = Fi(conv)
    sph.parameters.use_hydro_flag = True
    sph.parameters.radiation_flag = False
    sph.parameters.gamma = 1
    sph.parameters.isothermal_flag = True
    sph.parameters.integrate_entropy_flag = False
    sph.parameters.timestep = dt  
    sph.parameters.verbosity = 0
    sph.parameters.eps_is_h_flag = False    # h_smooth is constant
    eps = 0.1 | units.parsec
    sph.parameters.gas_epsilon = eps
    sph.parameters.sph_h_const = eps
    parts.h_smooth= eps

    print('eps-h flag', sph.get_eps_is_h(), sph.get_consthsm())

    expected_dt = 0.2*numpy.pi*numpy.power(eps, 1.5) \
                   / numpy.sqrt(constants.G*Mcloud/N)

    print("dt_exp=", expected_dt.value_in(units.Myr))
    print("dt=", dt)
    print("eps=", sph.parameters.gas_epsilon.in_(units.parsec))

    sph.gas_particles.add_particles(parts)

    channel_from_sph_to_parts= sph.gas_particles.new_channel_to(parts)
    channel_from_parts_to_sph= parts.new_channel_to(sph.gas_particles)
    
    i = 0
    L = 2.4*half_length.value_in(units.parsec)  # map covers the whole cloud
    # Colour scale around the mean cloud density
    log_rho = numpy.log10(rho_cloud.value_in(units.amu/units.cm**3))
    vmin, vmax = log_rho-2, log_rho+1
    E0 = 0.0
    ttarget = 0.0 | units.Myr

    plot_hydro(ttarget, sph, i, L, vmin, vmax, prefix)

    while ttarget < t_end:
        ttarget = float(i)*dt
        print(ttarget)
        sph.evolve_model(ttarget, timestep=dt)
        E = sph.gas_particles.kinetic_energy() \
             + sph.gas_particles.potential_energy() \
             + sph.gas_particles.thermal_energy()
        E_th = sph.gas_particles.thermal_energy()
        if i == 0:
            E0 = E
        Eerr = (E-E0)/E0
        print('energy=', E, 'energy_error=', Eerr, 'e_th=', E_th)
        channel_from_sph_to_parts.copy()
        plot_hydro(ttarget, sph, i, L, vmin, vmax, prefix)
        plot_cloud_volume(parts, L=L, vmin=vmin, vmax=vmax, i=i, prefix=prefix)
        i += 1

    sph.stop()
    return parts