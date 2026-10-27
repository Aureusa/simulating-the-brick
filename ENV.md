# Quick start
On a compute node, everything below is done by:
```bash
source amuse_env.sh
python3 brick/create_cloud.py
```

# Required Modules

```bash
module load ALICE/default

module load Miniconda3/24.7.1-0

module load AMUSE/2025.2.0-CUDA

# Compute nodes lack the system libfftw3 that the Fi worker links against
module load FFTW/3.3.10-GCC-13.2.0
```

# Running on a compute node (srun / sbatch) - going around MPI errors
Inside a Slurm job, OpenMPI tries to launch AMUSE workers through Slurm and
fails with `MPI_ERR_SPAWN: could not spawn processes`. Hide the Slurm
environment from OpenMPI before running python:
```bash
for v in $(env | grep -oE "^(SLURM|OMPI_MCA_plm)_[A-Za-z0-9_]*"); do unset $v; done
# The default PSM3 transport crashes the worker (ptl.c assertion); use shared memory/TCP
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp
python create_cloud.py
```

# Activate venv
```bash
conda activate amuse_env
```
