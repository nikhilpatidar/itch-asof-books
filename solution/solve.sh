#!/bin/bash
set -euo pipefail
cp /solution/bookd.cpp /app/src/bookd.cpp
make -C /app
