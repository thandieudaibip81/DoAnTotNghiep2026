#!/bin/bash
# build.sh — Build Docker image for Fraud Detection Webapp
# Usage: ./build.sh [tag]
#   Example: ./build.sh v1.0
#            ./build.sh latest

set -e

TAG="${1:-latest}"
IMAGE_NAME="thandieudaibip/fraud-detection-webapp"

# Change directory to the project root
cd "$(dirname "$0")/.."

echo "=================================================="
echo "  Fraud Detection — Docker Build"
echo "  Image: ${IMAGE_NAME}:${TAG}"
echo "=================================================="

echo "🐳 Building Docker image..."
# Use "Machine Learning" as the build context so it has access to both webapp and models
docker build --platform linux/amd64 -f Ops/Dockerfile -t ${IMAGE_NAME}:${TAG} "Machine Learning"

echo "=================================================="
echo "  ✅ Build complete!"
echo "  Image: ${IMAGE_NAME}:${TAG}"
echo ""
echo "  Push to Docker Hub:"
echo "    docker push ${IMAGE_NAME}:${TAG}"
echo "=================================================="
