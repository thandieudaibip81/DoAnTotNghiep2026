pipeline {
    agent {
        kubernetes {
            workspaceVolume persistentVolumeClaimWorkspaceVolume(claimName: 'jenkins-workspace-pvc', readOnly: false)
            yaml '''
apiVersion: v1
kind: Pod
spec:
  serviceAccountName: jenkins-admin
  nodeSelector:
    kubernetes.io/hostname: node3
  containers:
    - name: docker
      image: docker:27-dind
      resources:
        requests:
          cpu: "500m"
          memory: "1Gi"
        limits:
          cpu: "2000m"
          memory: "3Gi"
      securityContext:
        privileged: true
      env:
        - name: DOCKER_TLS_CERTDIR
          value: ""
        - name: DOCKER_DRIVER
          value: "overlay2"
      volumeMounts:
        - name: docker-storage
          mountPath: /var/lib/docker
    - name: kubectl
      image: alpine/k8s:1.32.4
      command: ['sleep']
      args: ['infinity']
  volumes:
    - name: docker-storage
      persistentVolumeClaim:
        claimName: jenkins-docker-pvc
'''
        }
    }

    environment {
        DOCKER_IMAGE = 'thandieudaibip/fraud-detection-webapp'
        DOCKER_TAG   = "${env.BUILD_NUMBER}"
        KUBE_NS      = 'fraud-detection'
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/thandieudaibip81/DoAnTotNghiep2026.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                container('docker') {
                    sh '''
                        # Wait for Docker daemon to be ready (up to 5 minutes)
                        RETRIES=0
                        MAX_RETRIES=60
                        while ! docker info > /dev/null 2>&1; do
                            RETRIES=$((RETRIES+1))
                            if [ $RETRIES -ge $MAX_RETRIES ]; then
                                echo "Docker daemon failed to start after ${MAX_RETRIES} retries!"
                                exit 1
                            fi
                            echo "Waiting for Docker daemon... (attempt ${RETRIES}/${MAX_RETRIES})"
                            sleep 5
                        done
                        echo "Docker daemon is ready!"

                        # Configure Docker daemon with registry mirrors for better connectivity
                        mkdir -p /etc/docker
                        cat > /etc/docker/daemon.json <<'DAEMONJSON'
{
    "registry-mirrors": ["https://mirror.gcr.io"],
    "dns": ["8.8.8.8", "8.8.4.4", "1.1.1.1"],
    "max-concurrent-downloads": 3
}
DAEMONJSON

                        # Build the image with retry logic (up to 3 attempts)
                        BUILD_SUCCESS=false
                        for attempt in 1 2 3; do
                            echo "=== Build attempt ${attempt}/3 ==="
                            if docker build \
                                --platform linux/amd64 \
                                --network host \
                                -f Ops/Dockerfile \
                                -t ${DOCKER_IMAGE}:${DOCKER_TAG} \
                                -t ${DOCKER_IMAGE}:latest \
                                "Machine Learning"; then
                                BUILD_SUCCESS=true
                                echo "Build succeeded on attempt ${attempt}!"
                                break
                            else
                                echo "Build failed on attempt ${attempt}, waiting 15s before retry..."
                                sleep 15
                            fi
                        done

                        if [ "$BUILD_SUCCESS" != "true" ]; then
                            echo "All 3 build attempts failed!"
                            exit 1
                        fi
                    '''
                }
            }
        }

        stage('Push to Docker Hub') {
            steps {
                container('docker') {
                    withCredentials([usernamePassword(
                        credentialsId: 'dockerhub-credentials',
                        usernameVariable: 'DOCKER_USER',
                        passwordVariable: 'DOCKER_PASS'
                    )]) {
                        sh '''
                            echo "${DOCKER_PASS}" | docker login -u "${DOCKER_USER}" --password-stdin

                            # Push with retry logic
                            for attempt in 1 2 3; do
                                echo "=== Push attempt ${attempt}/3 ==="
                                if docker push ${DOCKER_IMAGE}:${DOCKER_TAG} && \
                                   docker push ${DOCKER_IMAGE}:latest; then
                                    echo "Push succeeded on attempt ${attempt}!"
                                    break
                                else
                                    echo "Push failed on attempt ${attempt}, waiting 10s..."
                                    sleep 10
                                fi
                            done
                        '''
                    }
                }
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                container('kubectl') {
                    sh """
                        kubectl set image deployment/fraud-guard-webapp \
                            webapp=${DOCKER_IMAGE}:${DOCKER_TAG} \
                            -n ${KUBE_NS}
                        kubectl rollout status deployment/fraud-guard-webapp \
                            -n ${KUBE_NS} --timeout=300s
                    """
                }
            }
        }
    }

    post {
        success {
            echo " Pipeline thành công! Image: ${DOCKER_IMAGE}:${DOCKER_TAG}"
        }
        failure {
            echo " Pipeline thất bại! Kiểm tra log để biết chi tiết."
        }
    }
}
