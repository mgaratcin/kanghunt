*This program is not functioning yet, or ready for use.
Updates will be provided when it is fully functional.*
```  
nvcc -O3 -o mltbin mltbin.cu -std=c++11 -lpthread
g++ -O3 -o validate_points validate_points.cpp
```
On Debian based systems, run this commands to update your current enviroment
and install the tools needed to compile it 

```
apt install libssl-dev -y
apt install libgmp-dev -y
git clone https://github.com/bitcoin-core/secp256k1.git
cd secp256k1
sudo apt-get update
sudo apt-get install autoconf libtool pkg-config build-essential libgmp-dev
./autogen.sh
./configure --enable-module-recovery --enable-module-ecdh
make
sudo make install
sudo ldconfig
ls /usr/local/include/secp256k1_ecdh.h
```
