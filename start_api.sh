#!/bin/bash
# Start Credit Risk API with required environment variables

export DYLD_LIBRARY_PATH="/usr/local/opt/libomp/lib:$DYLD_LIBRARY_PATH"
python api/main.py
