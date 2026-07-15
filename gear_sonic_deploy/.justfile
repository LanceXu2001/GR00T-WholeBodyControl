# Like GNU `make`, but `just` rustier.
# https://just.systems/
# run `just` from this directory to see available commands

alias b := build
alias bd := build-debug
alias r := run
alias rd := run-debug
alias t := test
alias c := clean
alias ch := check

# Default command when 'just' is run without arguments
default:
  @just --list

# Get the number of cores
CORES := if os() == "macos" { `sysctl -n hw.ncpu` } else if os() == "linux" { `nproc` } else { "1" }

# Build the project (default: Release)
build *build_type='Release':
  @mkdir -p build
  @echo "Configuring the build system ({{build_type}})..."
  @cd build && cmake -S .. -B . -DCMAKE_BUILD_TYPE={{build_type}} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  @echo "Building the project..."
  @cd build && cmake --build . -j{{CORES}}
  @if [ "{{build_type}}" = "Debug" ]; then \
    echo "✅ Build complete! Executable: target/debug/g1_deploy_onnx_ref"; \
  else \
    echo "✅ Build complete! Executable: target/release/g1_deploy_onnx_ref"; \
  fi

# Build Debug version (convenience alias)
build-debug:
  @just build Debug

# Run a package (default: Release)
# Usage: just run g1_deploy_onnx_ref arg1 arg2 --flag value
#        just run [package] [args...]  # package defaults to 'g1_deploy_onnx_ref'
# Note: All arguments after package name are passed to the executable
run package='g1_deploy_onnx_ref' *args:
  @./target/release/{{package}} {{args}}

# Run Debug version
# Usage: just run-debug g1_deploy_onnx_ref arg1 arg2 --flag value
#        just run-debug [package] [args...]  # package defaults to 'g1_deploy_onnx_ref'
# Note: All arguments after package name are passed to the executable
run-debug package='g1_deploy_onnx_ref' *args:
  @./target/debug/{{package}} {{args}}

# Run code quality tools
test:
  @echo "Running tests..."

# Remove build artifacts and non-essential files
clean:
  @echo "Cleaning..."
  @rm -rf build
  @rm -rf target

# Run code quality tools
check:
  @echo "Running code quality tools..."
  @cppcheck --error-exitcode=1 --project=build/compile_commands.json -i build/_deps/

