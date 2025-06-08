{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-25.05.tar.gz") {} }:

let
  # Custom VkFFT package (header-only library)
  vkfft = pkgs.stdenv.mkDerivation rec {
    pname = "VkFFT";
    version = "1.3.4";
    
    src = pkgs.fetchFromGitHub {
      owner = "DTolm";
      repo = "VkFFT";
      rev = "v${version}";
      sha256 = "sha256-v23sLMlcv71P9qcNMH+NnCroI7GOks90PGBS3tR1RUs=";
    };

    buildInputs = with pkgs; [
      cudatoolkit
      # ocl-icd
      # opencl-headers
    ];
    
  # cmakeFlags = [
  #   "-DVKFFT_BACKEND=1"
  #   # "-DCMAKE_CUDA_COMPILER=${pkgs.cudatoolkit.bin}/bin/nvcc"
  # ];

  installPhase = ''
      mkdir -p $out/include
      cp -r vkFFT/ $out/include/
    '';
  };

  # Custom pyvkfft package
  pyvkfft = pkgs.python312Packages.buildPythonPackage rec {
    pname = "pyvkfft";
    version = "2024.1.4";
    
    src = pkgs.fetchFromGitHub {
      owner = "vincefn";
      repo = "pyvkfft";
      rev = "${version}";
      sha256 = "sha256-GwFu+Rlkek2dHdI8CZDovUCpWYwZ6Br++F6W33eXGzE=";
      fetchSubmodules = true;
    };
    
    nativeBuildInputs = with pkgs; [
      pkg-config
      cudatoolkit
      cudatoolkit.lib
      gcc12
    ];

    buildInputs = with pkgs; [
      vkfft
      cudatoolkit
      cudatoolkit.lib
      ocl-icd
      opencl-headers
      cudaPackages.cuda_cudart
      cudaPackages.cuda_nvrtc
    ];
    
    propagatedBuildInputs = with pkgs.python312Packages; [
      numpy
      pyopencl
      pycuda
    ];
    
    # Set environment variables for the build
    preBuild = ''
      export VKFFT_BACKEND=cuda,opencl
      export CPATH="${vkfft}/include:$CPATH"
      export CC=${pkgs.gcc12}/bin/gcc
      export CXX=${pkgs.gcc12}/bin/g++
      export HOST_COMPILER=${pkgs.gcc12}/bin/g++
      # export NVCCFLAGS="-arch=sm_89"
    '';

    installPhase = ''
      mkdir -p $out/${pkgs.python312.sitePackages}/pyvkfft
      cp -r pyvkfft/* $out/${pkgs.python312.sitePackages}/pyvkfft/
      cp -r build/lib*/pyvkfft/*.so $out/${pkgs.python312.sitePackages}/pyvkfft/
    '';    

    # Skip tests during build (they require GPU)
    doCheck = false;
    
    meta = with pkgs.lib; {
      description = "Python interface to VkFFT";
      homepage = "https://github.com/vincefn/pyvkfft";
      license = licenses.mit;
    };
  };

in pkgs.mkShell {
  nativeBuildInputs = with pkgs; [ pkg-config ];  
  buildInputs = with pkgs; [
    ruff
    black
    python312
    freetype
    glfw
    vkfft
    (python312Packages.python.withPackages (ps: with ps; [
      ffmpeg
      matplotlib
      numpy
      opencv4
      plotly
      psutil
      scipy
      torch-bin
      pytest
      pyopencl
      pyvkfft  # Add our custom pyvkfft package
      pycuda
      siphash24
    ]))
    cudatoolkit
    ocl-icd
    opencl-headers
  ];

  shellHook = ''
    echo "python environment with CUDA and pyvkfft."
    export TMPDIR=/tmp
  '';
}
