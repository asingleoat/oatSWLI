{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python311
    (python311Packages.python.withPackages (ps: with ps; [ numpy opencv4 matplotlib ]))
  ];

  shellHook = ''
    echo "Python environment with OpenCV and NumPy is ready."
  '';
}
