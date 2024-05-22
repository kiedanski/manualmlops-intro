docker build --platform linux/amd64 -t train_docker -f Dockerfile_train .

docker tag train_docker northamerica-northeast1-docker.pkg.dev/mlops-devrel/train/train:latest
docker push northamerica-northeast1-docker.pkg.dev/mlops-devrel/train/train:latest

