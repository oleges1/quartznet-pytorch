# never tested
docker build --file Dockerfile --tag quartznet-pytorch .
docker run -it --name=quartznet --runtime=nvidia --ipc=host -p 80:80 -p 8888:8888 -p 6006:6006 -v ~/:~/ quartznet-pytorch:latest bash
