# Usage
first make sure you have the required python dependencies. if you're using Nix, you can just enter a nix-shell with:
❯ nix-shell

otherwise you can inspect the dependencies in the shell.nix file and use your preferred method (TODO: add requirements.txt or its UV equivalent or whatever non-Nix python devs are using these days, poetry??).

❯ python3 accelerated.py ringgage_100fps_500nmps.AVI --gpu

tested and developed on linux, testing on Macos is todo, and windows is a maybe as I don't have a windows machine
