#!/bin/bash
# Train Credit Risk models with required environment variables

export DYLD_LIBRARY_PATH="/usr/local/opt/libomp/lib:$DYLD_LIBRARY_PATH"
python train_pipeline.py
