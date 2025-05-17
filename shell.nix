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
      ffmpeg
      matplotlib
      numpy
      opencv4
      plotly
      psutil
      scipy
      torch-bin
    ]))
    cudatoolkit  # Ensures CUDA libraries are available
  ];

  shellHook = ''
    echo "python environment with CUDA."
    export TMPDIR=/tmp # so that things like `eval (ssh-agent -c)` will work in the nix-shell
  '';
}
