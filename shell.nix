let
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-25.05.tar.gz") {
    config = {
      cudaSupport = true;
      allowUnfree = true;
    };
  };
  cuda = pkgs.cudaPackages_12_8;
in
  pkgs.mkShell {
    packages = with pkgs; [
      mpi
      libyaml
      gcc

      cudaPackages.nccl

      # runtime cuda libraries
      cuda.cuda_nvcc
      cuda.cudatoolkit
      cuda.cuda_nvrtc
      cuda.cudnn
      cuda.nccl

      ucx

      imagemagick

      # for jupyter notebook (basics, for more advanced use jupyenv)
      (python312.withPackages (pp:
        with pp; [
          dask
          distributed

          ray

          numba

          numpy
          cupy

          jax
          jaxlib
          (pp.callPackage ./mpi4jax.nix {inherit pkgs;})

          torch-bin

          mpi4py

          # for jupyter notebook (basics, for more advanced use jupyenv)
          ipython
          jupyter

          pillow
          tqdm
          ipywidgets
          imageio
          matplotlib
        ]))
    ];

    env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc
      pkgs.zlib
      cuda.cudatoolkit
      cuda.cuda_nvrtc
      cuda.cudnn
      cuda.nccl
      pkgs.mpi
    ];
    shellHook = ''
      # export LD_LIBRARY_PATH=${cuda.cudatoolkit}/lib:${cuda.cuda_nvrtc}/lib:${cuda.cudnn}/lib:${cuda.nccl}/lib:$LD_LIBRARY_PATH
      export CUDA_PATH=${cuda.cudatoolkit}
      export CUDA_HOME=$CUDA_PATH
      export OMP_NUM_THREADS=1
      export MKL_NUM_THREADS=1
      export OPENBLAS_NUM_THREADS=1
      export NUMBA_NUM_THREADS=1
    '';
  }
