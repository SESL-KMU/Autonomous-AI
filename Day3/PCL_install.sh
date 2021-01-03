sudo apt-get update

sudo apt-get install -y git
sudo apt-get install -y --no-install-recommends make cmake cmake-gui build-essential

sudo apt-get install -y git build-essential linux-libc-dev
sudo apt-get install -y libpcap-dev libproj-dev 
sudo apt-get install -y libusb-1.0-0-dev libusb-dev libudev-dev
sudo apt-get install -y mpi-default-dev openmpi-bin openmpi-common  
sudo apt-get install -y libflann1.8 libflann-dev
sudo apt-get install -y libeigen3-dev
sudo apt-get install -y libboost-all-dev
sudo apt-get install -y libvtk5.10-qt4 libvtk5.10 libvtk5-dev
sudo apt-get install -y libqhull* libgtest-dev
sudo apt-get install -y freeglut3-dev pkg-config
sudo apt-get install -y libxmu-dev libxi-dev 
sudo apt-get install -y mono-complete
sudo apt-get install -y qt-sdk openjdk-8-jdk openjdk-8-jre

git clone https://github.com/PointCloudLibrary/pcl.git -b pcl-1.7.2

cd pcl
mkdir release
cd release
cmake -DCMAKE_BUILD_TYPE=None -DCMAKE_INSTALL_PREFIX=/usr \
           -DBUILD_GPU=ON -DBUILD_apps=ON -DBUILD_examples=ON \
           -DCMAKE_INSTALL_PREFIX=/usr ..
make

sudo make install

sudo apt-get install -y ros-kinetic-pcl-ros





  

