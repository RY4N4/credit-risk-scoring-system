#!/bin/bash
# Start Credit Risk Frontend UI

echo "🚀 Starting Credit Risk Scoring Frontend..."
echo ""
echo "Make sure the API server is running first:"
echo "  ./start_api.sh"
echo ""
echo "Opening browser at http://localhost:8501"
echo ""

export DYLD_LIBRARY_PATH="/usr/local/opt/libomp/lib:$DYLD_LIBRARY_PATH"
streamlit run app.py
