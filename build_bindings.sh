#!/bin/bash
# Builds the lob_engine Python extension module from the C++ order book.
#
# Requires: pip3 install pybind11
#
# Usage: ./build_bindings.sh
# Produces: lob_engine<platform-suffix>.so in the project root, importable
# from Python as `import lob_engine` when run from this directory.

set -e  # stop immediately if any step fails, rather than limping onward

PYBIND_INCLUDE=$(python3 -c "import pybind11; print(pybind11.get_include())")
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
EXT_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

echo "pybind11 include: $PYBIND_INCLUDE"
echo "python include:   $PYTHON_INCLUDE"
echo "output suffix:    $EXT_SUFFIX"

# -undefined dynamic_lookup is needed on macOS (clang) so the Python
# symbols the extension needs are resolved at import time, not link time.
# On Linux this flag isn't needed and is harmlessly ignored by some
# compilers, but g++ on Linux doesn't accept it -- so we branch on OS.
if [[ "$OSTYPE" == "darwin"* ]]; then
    EXTRA_FLAGS="-undefined dynamic_lookup"
else
    EXTRA_FLAGS=""
fi

g++ -std=c++17 -O2 -Wall -shared -fPIC \
    -Iinclude -I"$PYBIND_INCLUDE" -I"$PYTHON_INCLUDE" \
    $EXTRA_FLAGS \
    bindings/lob_bindings.cpp src/order_book.cpp src/lobster_parser.cpp src/replay.cpp \
    -o "lob_engine${EXT_SUFFIX}"

echo "Built lob_engine${EXT_SUFFIX}"
echo "Test it: python3 tests/test_bindings.py"
