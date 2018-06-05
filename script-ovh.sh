sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install git virtualenv openjdk-8-jre-headless tmux -y
wget https://piccolo.link/sbt-1.1.6.tgz
tar -xvzf sbt-1.1.6.tgz
mkdir Soft
mkdir Data
cd Soft
git clone https://github.com/jtourille/yaset.git
git clone https://github.com/a-tsioh/MSETagger.git
cd
virtualenv -p python3 venv3
virtualenv -p python2 venv2

