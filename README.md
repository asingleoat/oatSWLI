# Usage

## Environment Setup

### Using Nix

With direnv set up, environment activation and teardown is automatic when
entering and leaving the project directory using the provided `.envrc` file,
following a one-time setup step:

  ```zsh
  # Allow the direnv configuration
  direnv allow .
  ```

The flake provides a consistent development environment with all dependencies
configured properly, including CUDA support.

### Manual Setup

If you're not using Nix, inspect the dependencies in `flake.nix` and install
them using your preferred method (TODO: add requirements.txt or its UV
equivalent or whatever non-Nix python devs are using these days, poetry??).

## Running the Code

You can use the included Makefile to download example data and run the code:

  ```zsh
  # Download example file and run with GPU acceleration
  make run
  
  # Only download the example file (4.3GB)
  make download
  
  # Clean up downloaded example file
  make clean
  ```

Or run the script directly:

  ```zsh
  python3 accelerated.py data/examples/ringgage_100fps_500nmps.AVI --gpu
  ```

Tested and developed on Linux, testing on MacOS is todo, and Windows is a maybe
as I don't have a Windows machine.

## License

This project is licensed under the terms of the GNU General Public License v3.0.
See [LICENSE](./gpl-3.0.txt) for details. I'm happy to consider requests for
more permissive licensing, find me on Twitter/X as asingleoat or email
asingleoat on gmail.
