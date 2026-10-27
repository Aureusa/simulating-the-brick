from amuse.units import units

MASS = 10e5 | units.MSun
RADIUS = 2.5 | units.parsec # Radius of the sphere before it is shaped (equal-volume radius)

DENSITY = MASS / (4/3 * 3.14159 * RADIUS**3) # Solar Mass per cubic parsec
NUMBER_OF_PARTICLES = 1000 | units.none

# Shape of the Brick (G0.253+0.016): an elongated, slightly curved filament
ASPECT_RATIO = 3.0 # Length / width
BEND = 1.0 | units.parsec # Sagitta: how far the middle sits off the line through the two ends

# Irregularities: smooth random displacement on top of the shape (0 pc = smooth cloud)
IRREGULARITY = 0.4 | units.parsec # rms displacement
IRREGULARITY_MIN_WAVELENGTH = 1.0 | units.parsec # smallest lumps
IRREGULARITY_MAX_WAVELENGTH = 8.0 | units.parsec # largest wiggles
IRREGULARITY_SEED = 42 # change for a different random pattern
