import numpy
  
from matplotlib import pyplot 

from amuse.lab import *
from amuse.ext.molecular_cloud import molecular_cloud
from amuse.ext.evrard_test import body_centered_grid_unit_cube

import properties as prop


def smooth(grid, s):
    """
    Gaussian smoothing of a 2D or 3D grid with a width of s pixels,
    done separately along each axis.
    """
    k = numpy.arange(-int(4*s)-1, int(4*s)+2)
    kernel = numpy.exp(-0.5*(k/s)**2)
    kernel /= kernel.sum()
    for axis in range(grid.ndim):
        grid = numpy.apply_along_axis(numpy.convolve, axis, grid,
                                      kernel, mode="same")
    return grid

def density_grid(parts, L=None, N=64, smoothing=0.2):
    """
    Smoothed 3D density of the particles in amu/cm^3 on an N^3 grid of size
    L [pc] (default: just enclosing the cloud), indexed [x, y, z].
    Returns the positions [pc], L, the cell edges [pc], the cell size [cm]
    and the density grid.
    """
    pc = units.parsec
    pos = numpy.array([parts.x.value_in(pc), parts.y.value_in(pc),
                       parts.z.value_in(pc)]).T
    m = parts.mass.value_in(units.amu)
    if L is None:
        L = 2.2*numpy.abs(pos).max()

    edges = numpy.linspace(-L/2, L/2, N+1)
    mass, _ = numpy.histogramdd(pos, bins=[edges]*3, weights=m)
    dx = (L/N | pc).value_in(units.cm)
    rho = smooth(mass/dx**3, smoothing/(L/N))
    return pos, L, edges, dx, rho

def shape_brick(parts, Rcloud, aspect_ratio, bend):
    """
    Turn a spherical cloud into an elongated, curved one like the Brick.
    Both steps conserve volume, so the mean density is unchanged.
    Returns the half-length of the cloud.
    """
    # Stretch along x and squeeze y and z so that length / width = aspect_ratio
    s = aspect_ratio**(2./3.)
    parts.x *= s
    parts.y /= numpy.sqrt(s)
    parts.z /= numpy.sqrt(s)

    # Bend in the x-y plane: shift y by a parabola in x (ends move by `bend`)
    half_length = s*Rcloud
    parts.y += bend*(parts.x/half_length)**2

    parts.move_to_center()
    return half_length

def add_irregularities(parts, amplitude, min_wavelength, max_wavelength,
                       n_modes=20, seed=42):
    """
    Displace the particles by a smooth random field: a sum of n_modes plane
    waves with random directions and wavelengths between min_wavelength and
    max_wavelength. This makes the edges lumpy, the width vary along the
    cloud and creates denser and emptier patches inside.
    `amplitude` is the rms displacement per coordinate.

    The effects of these pertubations should be studied by using
    different random seeds and varying the number of modes, amplitude, and wavelength range.
    """
    rng = numpy.random.default_rng(seed)
    pc = units.parsec
    pos = numpy.array([parts.x.value_in(pc), parts.y.value_in(pc),
                       parts.z.value_in(pc)]).T

    # Random wave vectors with wavelengths spread evenly in log
    wavelength = numpy.exp(rng.uniform(numpy.log(min_wavelength.value_in(pc)),
                                       numpy.log(max_wavelength.value_in(pc)),
                                       n_modes))
    direction = rng.normal(size=(n_modes, 3))
    direction /= numpy.linalg.norm(direction, axis=1)[:, None]
    k = 2*numpy.pi*direction/wavelength[:, None]
    phase = rng.uniform(0, 2*numpy.pi, size=(n_modes, 3))

    # Each sine has rms 1/sqrt(2), so this gives an rms of `amplitude`
    a = amplitude.value_in(pc)/numpy.sqrt(n_modes/2.)
    kr = pos @ k.T  # (N, n_modes)
    disp = a*numpy.sin(kr[:, :, None] + phase[None, :, :]).sum(axis=1)

    parts.x += disp[:, 0] | pc
    parts.y += disp[:, 1] | pc
    parts.z += disp[:, 2] | pc
    parts.move_to_center()

def plot_cloud_3d(parts, L=None, N=64, smoothing=0.2, prefix="GMC_"):
    """
    3D view of the cloud: the particles coloured by their local density, with
    the projected (column) density maps drawn on the three back walls.
    The density is estimated like in plot_cloud, but on an N^3 grid.
    """
    pos, L, edges, dx, rho = density_grid(parts, L, N, smoothing)

    # Local density at each particle: value of the cell it sits in
    idx = numpy.clip(((pos+L/2)/(L/N)).astype(int), 0, N-1)
    log_rho_p = numpy.log10(1.e-5+rho[idx[:, 0], idx[:, 1], idx[:, 2]])

    fig = pyplot.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")

    # Column density maps on the walls at x=-L/2, y=+L/2 and z=-L/2
    c = (edges[:-1]+edges[1:])/2
    A, B = numpy.meshgrid(c, c, indexing="ij")
    for axis, (u, v, w) in enumerate([("y", "z", "x"), ("x", "z", "y"),
                                      ("x", "y", "z")]):
        log_sigma = numpy.log10(1.e-5+rho.sum(axis=axis)*dx)
        vmax = log_sigma.max()
        levels = numpy.linspace(vmax-3, vmax, 20)
        offset = L/2 if w == "y" else -L/2
        if w == "x":
            ax.contourf(log_sigma, A, B, zdir="x", offset=offset,
                        levels=levels, cmap="jet")
        elif w == "y":
            ax.contourf(A, log_sigma, B, zdir="y", offset=offset,
                        levels=levels, cmap="jet")
        else:
            ax.contourf(A, B, log_sigma, zdir="z", offset=offset,
                        levels=levels, cmap="jet")

    # Particles, densest drawn last so they stay on top
    order = numpy.argsort(log_rho_p)
    sc = ax.scatter(pos[order, 0], pos[order, 1], pos[order, 2],
                    c=log_rho_p[order], cmap="jet", s=3, alpha=0.6,
                    depthshade=False)
    cbar = fig.colorbar(sc, orientation='vertical', fraction=0.03, pad=0.1)
    cbar.set_label('log local density [$amu/cm^3$]', rotation=270, labelpad=15)

    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-L/2, L/2)
    ax.set_zlim(-L/2, L/2)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [pc]")
    ax.set_ylabel("y [pc]")
    ax.set_zlabel("z [pc]")

    pyplot.savefig(prefix+"cloud_3d.png")
    pyplot.close(fig)

def plot_cloud_volume(parts, L=None, N=64, smoothing=0.2, n_surfaces=15,
                      vmin=None, vmax=None, i=None, prefix="GMC_"):
    """
    Volume rendering of the smoothed 3D density with plotly: a stack of
    semi-transparent density isosurfaces between vmin and vmax (log amu/cm^3).
    Without a frame number i it writes an interactive HTML file that can be
    rotated and zoomed in a browser. With i it writes the PNG frame
    <prefix>volume_<i>.png instead; pass a fixed L, vmin and vmax so that
    all frames of a movie share the same box and colour scale.
    Saving PNGs needs kaleido and a Chrome (see amuse_env.sh).
    """
    import plotly.graph_objects as go

    pos, L, edges, dx, rho = density_grid(parts, L, N, smoothing)
    log_rho = numpy.log10(1.e-5+rho)
    if vmax is None:
        vmax = log_rho.max()
    if vmin is None:
        vmin = vmax-2.5

    c = (edges[:-1]+edges[1:])/2
    # float32 halves the size of the HTML file
    X, Y, Z = numpy.meshgrid(c, c, c, indexing="ij")
    X, Y, Z, log_rho = (a.astype(numpy.float32) for a in (X, Y, Z, log_rho))
    fig = go.Figure(go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=log_rho.flatten(),
        isomin=vmin, isomax=vmax, surface_count=n_surfaces,
        opacity=0.1, colorscale="Jet",
        colorbar=dict(title="log density [amu/cm^3]")))
    fig.update_layout(scene=dict(xaxis_title="x [pc]", yaxis_title="y [pc]",
                                 zaxis_title="z [pc]", aspectmode="data"))

    if i is None:
        fig.write_html(prefix+"cloud_volume.html", include_plotlyjs=True)
    else:
        # Zero-padded so the frames sort in order for ffmpeg
        fig.write_image(prefix+"volume_%03d.png" % i, width=1280, height=960)

def plot_cloud(parts, L=None, N=200, smoothing=0.2, vmin=None, vmax=None,
               prefix="GMC_"):
    """
    Plot the projected (column) density of the cloud particles in the x-y plane,
    like plot_hydro but straight from the particles, without an SPH code.
    The particle masses are binned on an N x N grid of size L [pc] and smoothed
    with a Gaussian of width `smoothing` [pc], so each particle is spread out
    like an SPH kernel instead of showing up as a dot.
    """
    x = parts.x.value_in(units.parsec)
    y = parts.y.value_in(units.parsec)
    m = parts.mass.value_in(units.amu)
    if L is None:
        L = 2.2*max(numpy.abs(x).max(), numpy.abs(y).max())

    # Mass per pixel -> column density in amu/cm^2
    edges = numpy.linspace(-L/2, L/2, N+1)
    mass, _, _ = numpy.histogram2d(x, y, bins=[edges, edges], weights=m)
    dx = (L/N | units.parsec).value_in(units.cm)
    sigma = mass.T/dx**2  # transpose: imshow wants [y, x]

    sigma = smooth(sigma, smoothing/(L/N))

    log_sigma = numpy.log10(1.e-5+sigma)
    if vmax is None:
        vmax = log_sigma.max()
    if vmin is None:
        vmin = vmax-3

    fig = pyplot.figure()
    pyplot.xlabel("x [pc]")
    pyplot.ylabel("y [pc]")
    cax = pyplot.imshow(log_sigma, cmap="jet", origin="lower",
                        extent=[-L/2,L/2,-L/2,L/2], vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(cax, orientation='vertical', fraction=0.05)
    cbar.set_label('log column density [$amu/cm^2$]', rotation=270, labelpad=15)

    pyplot.savefig(prefix+"cloud.png")
    pyplot.close(fig)
