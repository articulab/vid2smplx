#!/bin/bash
# Quick demo: process a video with vid2smplx
#
# Usage:
#   bash examples/demo.sh /path/to/video.mp4
#
# This runs the full pipeline on the first 10% of the video for a quick test.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

VIDEO="${1:?Usage: bash examples/demo.sh /path/to/video.mp4}"

echo "Running vid2smplx demo on: $VIDEO"
echo "Processing first 10% of the video..."
echo ""

bash "$REPO_DIR/scripts/process_video.sh" "$VIDEO" \
    --percent 10

echo ""
echo "To run in production mode (NPZ only, no renders):"
echo "  bash scripts/process_video.sh $VIDEO --production"
