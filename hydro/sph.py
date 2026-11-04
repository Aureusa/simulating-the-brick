from properties import sph_prop as prop
from amuse.lab import Fi, units

EPS = prop.EPS

def setup_sph(converter, timestep: units.Myr):
    # Set up the SPH code with the desired parameters
    sph = Fi(converter)
    sph.parameters.use_hydro_flag = prop.USE_HYDRO_FLAG
    sph.parameters.radiation_flag = prop.RADIATION_FLAG
    sph.parameters.gamma = prop.GAMMA
    sph.parameters.isothermal_flag = prop.ISOTHERMAL_FLAG
    sph.parameters.integrate_entropy_flag = prop.INTEGRATE_ENTROPY_FLAG
    sph.parameters.timestep = timestep
    sph.parameters.verbosity = prop.VERBOSITY
    sph.parameters.eps_is_h_flag = prop.EPS_IS_H_FLAG    # h_smooth is constant
    sph.parameters.gas_epsilon = prop.GAS_EPSILON
    sph.parameters.sph_h_const = prop.SPH_H_CONST
    return sph
