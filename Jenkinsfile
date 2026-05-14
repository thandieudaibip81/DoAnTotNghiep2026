pipeline {
    agent {
        kubernetes {
            workspaceVolume persistentVolumeClaimWorkspaceVolume(claimName: 'jenkins-workspace-pvc', readOnly: false)
            yaml '''
apiVersion: v1
kind: Pod
spec:
  serviceAccountName: jenkins-admin
  containers:
    - name: docker
      image: docker:27-dind
      securityContext:
        privileged: true
      env:
        - name: DOCKER_TLS_CERTDIR
          value: ""
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
                        # Wait for Docker daemon to be ready
                        while ! docker info > /dev/null 2>&1; do
                            echo "Waiting for Docker daemon..."
                            sleep 2
                        done
                        echo "Docker daemon is ready!"

                        # Build the image
                        docker build \
                            --platform linux/amd64 \
                            -f Ops/Dockerfile \
                            -t ${DOCKER_IMAGE}:${DOCKER_TAG} \
                            -t ${DOCKER_IMAGE}:latest \
                            "Machine Learning"
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
                            docker push ${DOCKER_IMAGE}:${DOCKER_TAG}
                            docker push ${DOCKER_IMAGE}:latest
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
