# Set up AMUSE on ALICE. Use with: source amuse_env.sh
# Run on a compute node (srun/sbatch); see ENV.md for why each step is needed.

module load ALICE/default
module load Miniconda3/24.7.1-0
module load AMUSE/2025.2.0-CUDA
# Compute nodes lack the system libfftw3 that the Fi worker links against
module load FFTW/3.3.10-GCC-13.2.0

# Hide Slurm from OpenMPI so it can spawn AMUSE workers locally
for v in $(env | grep -oE "^(SLURM|OMPI_MCA_plm)_[A-Za-z0-9_]*"); do unset "$v"; done

# The default PSM3 transport crashes the worker; use shared memory/TCP
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp

# Chrome for plotly's PNG export (kaleido), downloaded with:
#   kaleido_get_chrome --path ~/.local/share/kaleido-chrome
export BROWSER_PATH=$HOME/.local/share/kaleido-chrome/chrome-linux64/chrome
# The X display forwarded to the login node does not exist on compute nodes
# and makes Chrome's WebGL fail
unset DISPLAY
