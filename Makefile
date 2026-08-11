.PHONY: all fleet local_autocov flying_conv

all: fleet local_autocov flying_conv
	@echo "All CUDA extensions installed."

local_autocov:
	@echo "Installing local_autocov..."
	cd local_autocov && pip install . --no-build-isolation

flying_conv:
	@echo "Installing flying_conv..."
	cd flying_conv && pip install . --no-build-isolation

fleet:
	@echo "Installing fleet..."
	cd FLEET && pip install . --no-build-isolation