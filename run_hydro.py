from brick import TheBrick
from hydro import run_hydro

if __name__ == "__main__":
    brick = TheBrick.from_properties()
    run_hydro(brick, code="Fi")
    