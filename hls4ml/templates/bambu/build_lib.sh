#!/usr/bin/env bash
set -euo pipefail

APPIMAGE="bambu"
FALLBACK_CC="g++"

TMPINFO=""
MOUNT_PID=""
MOUNT_DIR=""

cleanup() {
    if [ -n "${MOUNT_PID:-}" ]; then kill "$MOUNT_PID" 2>/dev/null || true; fi
    if [ -n "${TMPINFO:-}" ]; then rm -f "$TMPINFO" || true; fi
}
trap cleanup EXIT

# Try to mount to Bambu AppImage to access C++ compiler
if command -v "$APPIMAGE" >/dev/null 2>&1; then
    TMPINFO="$(mktemp)"
    "$APPIMAGE" --appimage-mount >"$TMPINFO" 2>&1 &
    MOUNT_PID=$!

    for _ in {1..100}; do
        if [ -s "$TMPINFO" ]; then
            MOUNT_DIR=$(sed -n '1p' "$TMPINFO" | tr -d '\r\n')
            if [ -d "$MOUNT_DIR" ]; then break; fi
        fi
        sleep 0.05
    done

    if [ ! -d "$MOUNT_DIR" ]; then
        MOUNT_DIR=""
        MOUNT_PID=""
    fi
fi

# If no Bambu AppImage: check for extracted AppImage (squashfs-root)
if [ -z "$MOUNT_DIR" ]; then
    BIN_PATH="$(which "$APPIMAGE" 2>/dev/null || true)"

    if [ -n "$BIN_PATH" ]; then
        APPDIR="$(dirname "$(dirname "$(dirname "$BIN_PATH")")")"

        if [ -x "$APPDIR/usr/bin/clang++-16" ] || \
           [ -x "$APPDIR/usr/compilers/clang-16/bin/clang++-16" ]; then
            MOUNT_DIR="$APPDIR"
        fi
    fi
fi

# If Bambu provides Clang++-16, use it
if [ -n "$MOUNT_DIR" ] && [ -x "$MOUNT_DIR/usr/bin/clang++-16" ]; then
    CC="$MOUNT_DIR/usr/bin/clang++-16"
    echo "Found clang++-16 in Bambu AppImage usr/bin directory."
elif [ -n "$MOUNT_DIR" ] && [ -x "$MOUNT_DIR/usr/compilers/clang-16/bin/clang++-16" ]; then
    CC="$MOUNT_DIR/usr/compilers/clang-16/bin/clang++-16"
    echo "Found clang++-16 in Bambu AppImage usr/compilers directory."
else
    echo "Bambu's clang++-16 not found. Using fallback compiler."
    CC="$FALLBACK_CC"
fi

echo "Using compiler: $($CC --version | head -n1)"

CFLAGS="-O3 -fPIC"

# Include -std=c++23 if the compiler supports it (enables half and bfloat16 types, errors otherwise)
if echo "" | $CC -Werror -fsyntax-only -std=c++23 -xc++ - -o /dev/null &>/dev/null; then
    CFLAGS+=" -std=c++23"
else
    CFLAGS+=" -std=c++14"
fi

# Include -fno-gnu-unique if it is there
if echo "" | $CC -Werror -fsyntax-only -fno-gnu-unique -xc++ - -o /dev/null &>/dev/null; then
    CFLAGS+=" -fno-gnu-unique"
fi

LDFLAGS=""
INCFLAGS="-Ifirmware/ac_types/include"
PROJECT="myproject"
LIB_STAMP="mystamp"
BASEDIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS_DIR="\"${BASEDIR}/firmware/weights\""

$CC $CFLAGS $INCFLAGS -D WEIGHTS_DIR="${WEIGHTS_DIR}" -c firmware/${PROJECT}.cpp -o ${PROJECT}.o
$CC $CFLAGS $INCFLAGS -D WEIGHTS_DIR="${WEIGHTS_DIR}" -c ${PROJECT}_bridge.cpp -o ${PROJECT}_bridge.o
$CC $CFLAGS $INCFLAGS -shared ${PROJECT}.o ${PROJECT}_bridge.o -o firmware/${PROJECT}-${LIB_STAMP}.so

rm -f *.o

echo "Done."
