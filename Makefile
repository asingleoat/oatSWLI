.PHONY: all run download clean setup

# Define variables
EXAMPLE_DIR = data/examples
EXAMPLE_FILE = $(EXAMPLE_DIR)/ringgage_100fps_500nmps.AVI
GDRIVE_ID = 1Ced8oHJ9SYO-Bt3TyE12kokPsH3x5XjK
SCRIPT = accelerated.py

all: download run

setup:
	@mkdir -p $(EXAMPLE_DIR)

download: setup
	@if test -f "$(EXAMPLE_FILE)"; then \
		echo "Example file already exists, skipping download."; \
	else \
		echo "Downloading example file (4.3GB)..."; \
		pip install gdown 2>/dev/null || echo "gdown already installed"; \
		gdown $(GDRIVE_ID) -O "$(EXAMPLE_FILE)"; \
		echo "Download complete."; \
	fi

run: download
	@echo "Running example with GPU acceleration..."
	python3 $(SCRIPT) $(EXAMPLE_FILE) --gpu

clean:
	@echo "Cleaning up downloaded files..."
	rm -f $(EXAMPLE_FILE)
