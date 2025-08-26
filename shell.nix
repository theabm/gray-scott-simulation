let
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-25.05.tar.gz") {
    config = {
      cudaSupport = true;
      allowUnfree = true;
    };
  };

  mpiWithCuda = pkgs.mpi.override {cudaSupport = true;};

  ucxWithCuda = pkgs.ucx.override {enableCuda = true;};

  cuda = pkgs.cudaPackages_12_8;
in
  pkgs.mkShell {
    packages = with pkgs; [
      mpiWithCuda
      libyaml
      gcc

      cudaPackages.nccl

      # runtime cuda libraries
      cuda.cuda_nvcc
      cuda.cudatoolkit
      cuda.cuda_nvrtc
      cuda.cudnn
      cuda.nccl

      ucxWithCuda

      # for jupyter notebook (basics, for more advanced use jupyenv)
      (python312.withPackages (pp: [
        pp.dask
        pp.distributed

        pp.ray

        pp.numpy
        pp.cupy

        pp.jax
        pp.torch-bin

        (pp.mpi4py.override {mpi = mpiWithCuda;})

        # for jupyter notebook (basics, for more advanced use jupyenv)
        pp.ipython
        pp.jupyter

        pp.pillow
        pp.tqdm
        pp.ipywidgets
        pp.imageio
        pp.matplotlib
      ]))
    ];
    shellHook = ''
      export CUDA_PATH=${cuda.cudatoolkit}
      export CUDA_HOME=$CUDA_PATH
      export LD_LIBRARY_PATH=${cuda.cudatoolkit}/lib:${cuda.cuda_nvrtc}/lib:${cuda.cudnn}/lib:${cuda.nccl}/lib:$LD_LIBRARY_PATH
      export NIX_CFLAGS_COMPILE = "-I${cuda.cudatoolkit}/include";
    '';
  }
