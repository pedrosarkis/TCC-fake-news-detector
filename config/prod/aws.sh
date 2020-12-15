# Instalar docker
sudo apt install -y curl apt-transport-https ca-certificates software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
sudo apt upgrade -y
apt-cache policy docker-ce
sudo apt install docker-ce docker-compose -y
sudo groupadd docker
sudo gpasswd -a $USER docker
sudo service docker restart

# Configurar deploy
sudo su ubuntu
cd /home/ubuntu
mkdir ./fake-news-detector.git
mkdir ./fake-news-detector
cd ./fake-news-detector.git
git init --bare
printf '#!/bin/sh\n\nGIT_WORK_TREE=/home/ubuntu/fake-news-detector git checkout -f
  cd /home/ubuntu/fake-news-detector
  docker-compose down && docker-compose up -d' > ./hooks/post-receive
chmod +x ./hooks/post-receive

# Configurar .env

# Localmente
git remote add aws fake-news:/home/ubuntu/fake-news-detector.git

# Inicializar
docker-compose -f docker-compose.yml -f docker-compose-prod.yml up -d