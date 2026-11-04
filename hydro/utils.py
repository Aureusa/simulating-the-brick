import numpy as np
from matplotlib import pyplot 
from amuse.lab import units


def make_map(sph, N=100, L=1):

    x,y = np.indices((N+1,N+1))

    x = L*(x.flatten()-N/2.)/N
    y = L*(y.flatten()-N/2.)/N
    z = x*0.
    vx = 0*x
    vy = 0*x
    vz = 0*x

    x = units.parsec(x)
    y = units.parsec(y)
    z = units.parsec(z)
    vx = units.kms(vx)
    vy = units.kms(vy)
    vz = units.kms(vz)

    rho,rhovx,rhovy,rhovz,rhoe = sph.get_hydro_state_at_point(x,y,z,vx,vy,vz)
    rho = rho.reshape((N+1,N+1))

    return rho


def plot_hydro(time, sph, i, L=10, vmin=2, vmax=5, prefix="GMC_"):
    fig = pyplot.figure()
    pyplot.xlabel("x [pc]")
    pyplot.ylabel("y [pc]")
    rho = make_map(sph,N=200,L=L)
    cax = pyplot.imshow(np.log10(1.e-5+rho.value_in(units.amu/units.cm**3)),
                        cmap="jet", origin="lower",
                        extent=[-L/2,L/2,-L/2,L/2],vmin=vmin,vmax=vmax)
    if i==5:
        cbar = fig.colorbar(cax, ticks=[vmin, vmax], orientation='vertical', fraction=0.05)
        cbar.set_label('log projected density [$amu/cm^3$]', rotation=270)
    
    pyplot.savefig(prefix+str(i)+".png")
    pyplot.close(fig)
    