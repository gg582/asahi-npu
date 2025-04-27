# LIB_NAME: The name of the library to be built.
LIB_NAME = libane

# CC: The C compiler to use.
CC = gcc

# CFLAGS: Compiler flags.
# -fPIC is required for shared libraries.
# Add both local and libdrm include paths.
CFLAGS = -I. -I/usr/include/libdrm -Wall -Werror -Wextra \
	-Wdeclaration-after-statement \
	-O3 -std=gnu99 -fPIC

# BUILD_DIR: Directory for build artifacts.
BUILD_DIR = .
# SRC_DIR: Directory for source files.
SRC_DIR = .

# Object file
OBJECT = $(BUILD_DIR)/ane.o

# Shared and static library targets
SHARED_LIB = $(BUILD_DIR)/${LIB_NAME}.so
STATIC_LIB = $(BUILD_DIR)/${LIB_NAME}.a

# Install directories
INCLUDE_DIR = /usr/include/${LIB_NAME}
LIB_DIR = /usr/lib64

.PHONY: all shared static install uninstall clean

# Build both shared and static libraries
all: shared static

# Build shared library
shared: $(OBJECT)
	$(CC) -shared -o $(SHARED_LIB) $(OBJECT) -L/usr/lib64 -ldrm

# Build static library
static: $(OBJECT)
	ar rcs $(STATIC_LIB) $(OBJECT)

# Compile .c to .o
$(BUILD_DIR)/%.o : $(SRC_DIR)/%.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Install headers and libraries
install: all
	install -d /usr/include/drm
	install -m 0644 ane_accel.h ane.h ane_f16.h /usr/include/drm
	install -d $(LIB_DIR)
	install -m 0755 $(SHARED_LIB) $(LIB_DIR)
	install -m 0644 $(STATIC_LIB) $(LIB_DIR)
	ldconfig

# Uninstall headers and libraries
uninstall:
	rm -rf /usr/include/drm
	rm -f $(LIB_DIR)/$(notdir $(SHARED_LIB))
	rm -f $(LIB_DIR)/$(notdir $(STATIC_LIB))
	ldconfig

# Clean build artifacts
clean:
	rm -f $(OBJECT) $(SHARED_LIB) $(STATIC_LIB)
