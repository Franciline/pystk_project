# PySTK camera-state patch

`pystk2-camera-state.patch` targets PySuperTuxKart2 0.7.4. It restores the
commented camera-state update in `pystk_cpp/state.cpp`, with guards for absent
cameras and scene nodes. It exposes the existing camera; it does not move it or
change the policy.

Source: https://pypi.org/project/PySuperTuxKart2/0.7.4/

Apply from the extracted source directory:

```bash
patch -p1 < /path/to/pystk_project/patches/pystk2-camera-state.patch
patch -p1 < /path/to/pystk_project/patches/pystk2-rescue-state.patch
patch -p1 < /path/to/pystk_project/patches/pystk2-headless-camera-guard.patch
```

The local test build lives in `/tmp/pystk-camera-build/build312`. The installed
virtual environment is unchanged. Use the test binding explicitly:

```bash
PYTHONPATH=/tmp/pystk-camera-build/build312 \
  ../.venv/bin/python src/main/plot_trajectory_screenshot.py \
  --model rlpd --step 385 --future-steps 10
```

The build directory is temporary. Rebuild from the patched source if it is
removed. Local CMake configuration used Ninja, Python 3.12, Release mode,
`CMAKE_PREFIX_PATH=/opt/homebrew`, `NO_SHADERC=ON`, `BUILD_RECORDER=OFF`,
`CHECK_ASSETS=OFF`, `USE_WIIUSE=OFF`, `USE_CRYPTO_OPENSSL=OFF`, and
`USE_SQLITE3=OFF`; build target `pystk2`.

Recreate the source download, checksum verification, patch and isolated build:

```bash
../.venv/bin/python src/main/build_camera_binding.py
```

This helper installs CMake and Ninja only into its temporary tools directory;
it does not install a replacement binding into the project virtual environment.
Existing system build dependencies (Apple command-line tools, SDL2, Freetype,
Harfbuzz, JPEG and PNG libraries) are required on macOS. An initial build takes
several minutes. `--build-root` chooses another location; update `PYTHONPATH`
to that location's `build312` directory when recording.

## Rescue diagnostics and headless map changes

The build helper also applies `pystk2-rescue-state.patch`, exposing the actual
rescue-animation status as `Kart.rescue_active`. It updates native pickling too,
so the value survives the Gymnasium subprocess connection. It identifies a
rescue, not whether its cause was a fall. Count transitions into this status.

`pystk2-headless-camera-guard.patch` skips camera-state updates without graphics;
this prevents stale camera-node access when headless training switches maps.
These changes do not add actor observations or alter executed actions. Original
virtual-environment binaries remain untouched. Use the same patched build for
offline collection, training and evaluation when collecting rescue diagnostics.
