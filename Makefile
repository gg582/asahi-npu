# SPDX-License-Identifier: GPL-2.0-only OR MIT
# Copyright 2023 Eileen Yoon <eyn@gmx.com>

# Set the kernel build directory
KERNELDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

# Object files to be compiled into the kernel module
obj-m := ane.o  # Final output module will be ane.ko

# Source files
ane-y := ane_drv.o ane_tm.o ane_onnx.o  # Compile driver, TM helper, and ONNX loader

# Default target: Build the kernel module
default:
	@echo "Building ANE kernel module..."
	@make -C $(KERNELDIR) M=$(PWD) modules

# Clean target: Remove build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@make -C $(KERNELDIR) M=$(PWD) clean

# Install target: Install the module to the kernel
install:
	@echo "Installing ANE module..."
	@sudo make -C $(KERNELDIR) M=$(PWD) modules_install

# Check target: Verify if CONFIG_DRM_ACCEL_ANE is enabled in the kernel config (Not needed in this case)
check:
	@echo "Checking kernel configuration for DRM_ACCEL_ANE..."
	@grep CONFIG_DRM_ACCEL_ANE /boot/config-$(shell uname -r)

# Additional target for building kernel modules directly
modules:
	@make -C $(KERNELDIR) M=$(PWD) modules

