{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-24.11.tar.gz") {} }:
# { pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  nativeBuildInputs = with pkgs; [ pkg-config ];  
  buildInputs = with pkgs; [
    ruff
    black
    python312
    freetype
    glfw
    (python312Packages.python.withPackages (ps: with ps; [
      numpy
      opencv4
      ffmpeg
      matplotlib
      torch-bin
      scipy
      plotly
    ]))
    cudatoolkit  # Ensures CUDA libraries are available
  ];

  shellHook = ''
    echo "python environment with CUDA."
  '';
}
