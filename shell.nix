let
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixpkgs-unstable.tar.gz") { };
in
pkgs.mkShell {
  packages = with pkgs; [
    pkgs.openmpi
    pkgs.libyaml
    pkgs.gcc

    # for jupyter notebook (basics, for more advanced use jupyenv)
    (pkgs.python3.withPackages (pp: [
      pp.dask
      pp.distributed
      pp.numpy

      pp.ray

      pp.mpi4py

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
}
