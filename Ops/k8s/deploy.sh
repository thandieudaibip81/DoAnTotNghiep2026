#!/bin/bash
# deploy.sh — Deploy Fraud Detection to Kubernetes
# Usage: ./deploy.sh
#
# Prerequisites:
#   - kubectl configured to connect to your K8s cluster
#   - Docker image already pushed to registry
#   - Update secret.yaml with real API keys

set -e

NAMESPACE="fraud-detection"

echo "=================================================="
echo "  Fraud Detection — K8s Deployment"
echo "=================================================="

# Step 1: Create namespace
echo ""
echo "📁 Step 1: Creating namespace..."
kubectl apply -f namespace.yaml

# Step 2: Create secrets and config
echo ""
echo "🔐 Step 2: Applying secrets and config..."
kubectl apply -f secret.yaml
kubectl apply -f configmap.yaml

# Step 3: Deploy PostgreSQL
echo ""
echo "🐘 Step 3: Deploying PostgreSQL..."
kubectl apply -f postgres.yaml
echo "   Waiting for PostgreSQL to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=120s

# Step 4: Deploy webapp
echo ""
echo "🚀 Step 4: Deploying webapp..."
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
echo "   Waiting for webapp to be ready..."
kubectl wait --for=condition=ready pod -l app=fraud-detection -n $NAMESPACE --timeout=120s

# Step 5: Show status
echo ""
echo "=================================================="
echo "  ✅ Deployment complete!"
echo "=================================================="
echo ""
echo "📊 Pod status:"
kubectl get pods -n $NAMESPACE -o wide
echo ""
echo "🌐 Service:"
kubectl get svc -n $NAMESPACE
echo ""
echo "🔗 Access the webapp at:"
echo "   http://192.168.10.110:30800"
echo "   http://192.168.10.111:30800"
echo "   http://192.168.10.112:30800"
echo ""
