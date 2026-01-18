.PHONY: all clean

GLSL_DIR = glsl
SPIRV_DIR = spirv

GLSL_SOURCES = $(wildcard $(GLSL_DIR)/*.comp)
SPIRV_TARGETS = $(patsubst $(GLSL_DIR)/%.comp,$(SPIRV_DIR)/%.spv,$(GLSL_SOURCES))

all: $(SPIRV_TARGETS)

$(SPIRV_DIR)/%.spv: $(GLSL_DIR)/%.comp
	@mkdir -p $(SPIRV_DIR)
	@echo "Compiling $< -> $@"
	@glslc $< -o $@

clean:
	rm -rf $(SPIRV_DIR)

info:
	@echo "GLSL sources: $(GLSL_SOURCES)"
	@echo "SPIR-V targets: $(SPIRV_TARGETS)"
