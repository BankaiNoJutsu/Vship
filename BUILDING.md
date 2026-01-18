# Building Vship (Rust/Vulkan)

Comprehensive build instructions for the Vship Rust & Vulkan rewrite.

## Prerequisites

### Required

1. **Rust** (1.70+)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Vulkan SDK**
   - **Linux (Ubuntu/Debian)**:
     ```bash
     sudo apt update
     sudo apt install vulkan-tools libvulkan-dev
     # For shader compilation
     sudo apt install glslc
     # Or install full Vulkan SDK from LunarG
     ```

   - **Linux (Arch)**:
     ```bash
     sudo pacman -S vulkan-devel shaderc
     ```

   - **macOS**:
     ```bash
     brew install vulkan-headers vulkan-loader molten-vk glslang
     ```

   - **Windows**:
     - Download and install [Vulkan SDK from LunarG](https://vulkan.lunarg.com/)
     - Add SDK bin directory to PATH

### Optional (for full features)

3. **FFmpeg Development Libraries** (for video file support)
   - **Linux (Ubuntu/Debian)**:
     ```bash
     sudo apt install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
     ```

   - **macOS**:
     ```bash
     brew install ffmpeg
     ```

   - **Windows**:
     - Download FFmpeg development files from [ffmpeg.org](https://ffmpeg.org/)
     - Set `FFMPEG_DIR` environment variable

## Build Steps

### 1. Clone Repository

```bash
git clone https://github.com/BankaiNoJutsu/Vship.git
cd Vship
git checkout claude/rewrite-vship-rust-vulkan-940VX
```

### 2. Compile Shaders

```bash
cd shaders
make
cd ..
```

This compiles GLSL shaders to SPIR-V bytecode in `shaders/spirv/`.

**Verify**: You should see `*.spv` files in `shaders/spirv/`:
- `rgb_to_xyb.spv`
- `gaussian_blur.spv`
- `downsample.spv`
- `ssim_error.spv`

### 3. Build Rust Workspace

#### Basic Build (without FFmpeg)

```bash
cargo build --release
```

#### Full Build (with FFmpeg support)

```bash
# Enable ffmpeg feature
cargo build --release --features ffmpeg
```

**Note**: FFmpeg integration is currently a placeholder. To fully enable:
1. Install FFmpeg dev libraries (see prerequisites)
2. Uncomment `ffmpeg-next` dependency in `ffvship/Cargo.toml`
3. Review `ffvship/src/ffmpeg_decoder.rs` for implementation notes

### 4. Run Tests

```bash
# Unit tests
cargo test

# Integration tests
cargo test --all

# Benchmarks
cargo bench
```

### 5. Build Individual Components

```bash
# Build core library
cargo build --release -p vship-core

# Build metrics library
cargo build --release -p vship-metrics

# Build C FFI library
cargo build --release -p vship-ffi

# Build CLI tool
cargo build --release -p ffvship
```

## Build Artifacts

After successful build, artifacts are located in `target/release/`:

- **Libraries**:
  - `libvship_core.rlib` - Core Vulkan abstraction
  - `libvship_metrics.rlib` - Metrics implementations
  - `libvship_ffi.so` / `.dylib` / `.dll` - C API shared library
  - `libvship_ffi.a` - C API static library

- **Executables**:
  - `ffvship` - CLI tool for video metrics

- **Headers**:
  - `include/vship.h` - Auto-generated C API header (created during build)

## Installation

### System-wide Installation

```bash
# Install CLI tool
cargo install --path ffvship

# Install C library (manual)
sudo cp target/release/libvship_ffi.so /usr/local/lib/
sudo cp include/vship.h /usr/local/include/
sudo ldconfig  # Linux only
```

### Local Installation

```bash
# Create local installation directory
mkdir -p ~/.local/vship

# Copy artifacts
cp target/release/ffvship ~/.local/vship/
cp target/release/libvship_ffi.* ~/.local/vship/
cp include/vship.h ~/.local/vship/

# Add to PATH
echo 'export PATH=$HOME/.local/vship:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$HOME/.local/vship:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Platform-Specific Notes

### Linux

- Ensure Vulkan drivers are installed for your GPU
- Check with: `vulkaninfo`
- For NVIDIA: Install `nvidia-vulkan-icd`
- For AMD: Install `mesa-vulkan-drivers`
- For Intel: Install `intel-media-va-driver-non-free`

### macOS

- Requires MoltenVK for Vulkan support
- macOS 10.13+ recommended
- Metal-capable GPU required

### Windows

- Vulkan SDK must be in PATH
- Visual Studio Build Tools may be required
- Use PowerShell or Git Bash for commands

## Troubleshooting

### Vulkan Not Found

```
Error: Failed to load Vulkan library
```

**Solution**: Install Vulkan SDK and ensure `VK_SDK_PATH` is set

### Shader Compilation Fails

```
glslc: command not found
```

**Solution**: Install Vulkan SDK with shader compiler (glslc)

### FFmpeg Not Found

```
warning: FFmpeg integration not enabled
```

**Solution**: This is expected if FFmpeg dev libraries aren't installed. The tool will work with placeholder video support. To enable full FFmpeg, see "Full Build" section above.

### GPU Not Detected

```
Error: No compatible Vulkan device found
```

**Solutions**:
1. Update GPU drivers
2. Verify with `vulkaninfo`
3. Check that GPU supports Vulkan 1.3+

### Build Fails on `gpu-allocator`

```
error: failed to compile gpu-allocator
```

**Solution**: Update Rust toolchain: `rustup update`

## Development Build

For development with faster compilation:

```bash
# Debug build (faster compilation, slower execution)
cargo build

# Check code without building
cargo check

# Format code
cargo fmt

# Lint code
cargo clippy

# Generate documentation
cargo doc --open
```

## Cross-Compilation

### For Windows (from Linux)

```bash
rustup target add x86_64-pc-windows-gnu
cargo build --release --target x86_64-pc-windows-gnu
```

### For ARM64 (Apple Silicon)

```bash
rustup target add aarch64-apple-darwin
cargo build --release --target aarch64-apple-darwin
```

## Continuous Integration

The project includes CI/CD configuration for:
- Automated testing
- Multi-platform builds
- Shader compilation verification
- Documentation generation

## Next Steps

- See [EXAMPLES.md](EXAMPLES.md) for usage examples
- See [README.md](README.md) for feature overview
- See [shaders/README.md](shaders/README.md) for shader documentation

## Support

- Report issues: https://github.com/BankaiNoJutsu/Vship/issues
- Original C++ version: https://codeberg.org/Line-fr/Vship
