*This program is not functioning yet, or ready for use.
Updates will be provided when it is fully functional.*
```  
nvcc -O3 -o mltbin mltbin.cu -std=c++11 -llz4 -lpthread
g++ -O3 -o validate_points validate_points.cpp
```
On Debian based systems, run this commands to update your current enviroment
and install the tools needed to compile it 

```
apt install libssl-dev -y
apt install libgmp-dev -y
```
