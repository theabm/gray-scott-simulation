{ pkgs ? import <nixpkgs> {} }:
let
  py = pkgs.python312Packages;
  mpiDev = pkgs.lib.getDev pkgs.mpi;  # <- the output that actually has mpicc
in
py.buildPythonPackage rec {
  pname = "mpi4jax";
  version = "0.8.1";
  format = "pyproject";

  src = pkgs.fetchPypi {
    inherit pname version;
    sha256 = "sha256-bVJDOZRS4/3kWdwos41SI7qxBK+wy375N51s0UQ1Bgo=";
  };

  nativeBuildInputs =
    (with py; [ setuptools wheel cython mpi4py ]) ++
    [ mpiDev ];   # <- ensure mpicc is on PATH during build

  propagatedBuildInputs = with py; [ mpi4py jax jaxlib ];

  # Make sure the build backend sees the correct mpicc path
  preBuild = ''
    export MPI4JAX_BUILD_MPICC="${mpiDev}/bin/mpicc"
    echo "Using MPI4JAX_BUILD_MPICC=$MPI4JAX_BUILD_MPICC"
    command -v mpicc || true
    "$MPI4JAX_BUILD_MPICC" --version || true
  '';

  strictDeps = true;
  doCheck = false;
  pythonImportsCheck = [ "mpi4jax" ];

  meta = with pkgs.lib; {
    description = "MPI primitives for JAX (synchronous and asynchronous)";
    homepage = "https://github.com/mpi4jax/mpi4jax";
    license = licenses.asl20;
  };
}
