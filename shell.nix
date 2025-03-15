{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-24.11.tar.gz") {} }:
# { pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  nativeBuildInputs = with pkgs; [ pkg-config ];  
  buildInputs = with pkgs; [
    python311
    cairo
    pkg-config
    freetype
    glfw
    (python311Packages.python.withPackages (ps: with ps; [
      numpy
      opencv4
      ffmpeg  # For video handling
      # cupy  # CuPy with CUDA acceleration
      matplotlib
      # torchWithCuda
      torch-bin
      scipy
      vispy
      pyglet
    ]))
    cudatoolkit  # Ensures CUDA libraries are available
  ];

  shellHook = ''
    echo "Python environment with CuPy and CUDA support is ready."
  '';
}
