*This program is not functioning yet, or ready for use.*
```
nvcc -o mltbin mltbin.cu -std=c++11
```
On Debian based systems, run this commands to update your current enviroment
and install the tools needed to compile it 

```
apt update && apt upgrade
apt install git -y
apt install build-essential -y
apt install libssl-dev -y
apt install libgmp-dev -y
sudo apt install liblz4-dev -y
wget https://github.com/Kitware/CMake/releases/download/v3.27.0/cmake-3.27.0.tar.gz
tar -zxvf cmake-3.27.0.tar.gz
cd cmake-3.27.0
./bootstrap
make
sudo make install
cmake --version
# Clone the NVCOMP repository
git clone https://github.com/NVIDIA/nvcomp.git
cd nvcomp

# Create a build directory and navigate into it
mkdir build && cd build

# Configure the build with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build and install NVCOMP
make -j$(nproc)
sudo make install

```
