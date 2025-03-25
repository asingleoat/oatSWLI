{
  description = "Python development environment with CUDA support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # Configure nixpkgs to allow unfree packages
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [ pkg-config ];  
          buildInputs = with pkgs; [
            python311
            freetype
            glfw
            (python311Packages.python.withPackages (ps: with ps; [
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
            echo "Python environment with CUDA support activated."
            echo "Note: This environment includes non-free software (CUDA, torch-bin)."
          '';
        };
      }
    );
} 