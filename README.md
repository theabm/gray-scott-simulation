This repo contains an implementation of the Gray Scott Solver and examples of how to couple the simulations to DOREISA.

# Gray Scott Simulation

A reaction-diffusion system described here that involves two generic chemical species U and V, whose concentration at a given point in space is referred to by variables u and v. As the term implies, they react with each other, and they diffuse through the medium. Therefore the concentration of U and V at any given location changes with time and can differ from that at other locations.

The overall behavior of the system is described by the following formula, two equations which describe three sources of increase and decrease for each of the two chemicals:


$$
\begin{array}{l}
\displaystyle \frac{\partial u}{\partial t} = D_u \Delta u - uv^2 + F(1-u) \\
\displaystyle \frac{\partial v}{\partial t} = D_v \Delta v + uv^2 - (F+k)v
\end{array}
$$

The laplacian is computed with the following numerical scheme

$$
\Delta u_{i,j} \approx u_{i,j-1} + u_{i-1,j} -4u_{i,j} + u_{i+1, j} + u_{i, j+1}
$$

The classic Euler scheme is used to integrate the time derivative.

# Folder Structure

- `python/` : 
    - contains 3 implementations of GS instrumented with MPI: `sim.py`, `sim-numba.py`, `sim-cupy.py`.
    - contains 1 implementation of GS + DOREISA: `sim-doreisa.py`.
    - [TODO] implementation of GS+DOREISA+GPU (cupy/torch/jax).
- `c/`: 
    - contains 1 implementation of GS instrumented with MPI: `sim.c`.
    - [TODO] contains 1 implementation of GS + DOREISA: `sim-doreisa.c`.
    - [TODO] implementation of GS+DOREISA+GPU (kokkos)
- `analytics`: 
    - contains 1 simple analytic which calculates the average of V and U on the fly: `avg.py`.
    - [TODO] analytics using deep neural networks.
- [TODO] `metrics`: contain metrics for automatic monitoring.
- `Makefile`: good entry point to give an idea of how to build and run the examples.

# Building

## C 

[TODO]

## Python

To run `sim-doreisa.py` you only need to create a virtual environment using the provided `requirements.txt`.
You can run `make py-install` to create the necessary venv. 

# Running

## C

[TODO]

## Python

Make sure the virtual environment is active, then simply run `bash launch-insitu-python.sh`. This will create a local ray instance, launch the analytics, launch the simulation, and then kill the ray instance.

## More Complex Examples

[TODO] Distributed example

## Nix
If you're using Nix, a shell.nix is provided for your own convenience at `nix/`. 

However, running the simulation + DOREISA puts some constrains on the Dask and Ray versions needed, so there is a `python/requirements.txt` which is needed to run `sim-doreisa.py`. 