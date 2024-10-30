nvcc -o mltbin mltbin.cu -std=c++11

On Debian based systems, run this commands to update your current enviroment
and install the tools needed to compile it 

```
apt update && apt upgrade
apt install git -y
apt install build-essential -y
apt install libssl-dev -y
apt install libgmp-dev -y
```

### Valid n and maximun k values for specific 

```
+------+----------------------+-------------+
| bits |  n in hexadecimal    | k max value |
+------+----------------------+-------------+
|   20 |             0x100000 | 1 (default) |
|   22 |             0x400000 | 2           |
|   24 |            0x1000000 | 4           |
|   26 |            0x4000000 | 8           |
|   28 |           0x10000000 | 16          |
|   30 |           0x40000000 | 32          |
|   32 |          0x100000000 | 64          |
|   34 |          0x400000000 | 128         |
|   36 |         0x1000000000 | 256         |
|   38 |         0x4000000000 | 512         |
|   40 |        0x10000000000 | 1024        |
|   42 |        0x40000000000 | 2048        |
|   44 |       0x100000000000 | 4096        |
|   46 |       0x400000000000 | 8192        |
|   48 |      0x1000000000000 | 16384       |
|   50 |      0x4000000000000 | 32768       |
|   52 |     0x10000000000000 | 65536       |
|   54 |     0x40000000000000 | 131072      |
|   56 |    0x100000000000000 | 262144      |
|   58 |    0x400000000000000 | 524288      |
|   60 |   0x1000000000000000 | 1048576     |
|   62 |   0x4000000000000000 | 2097152     |
|   64 |  0x10000000000000000 | 4194304     |
+------+----------------------+-------------+
```
### What values use according to my current RAM:

2 G
-k 128

4 G
-k 256

8 GB
-k 512

16 GB
-k 1024

32 GB
-k 2048

64 GB
-n 0x100000000000 -k 4096

128 GB
-n 0x400000000000 -k 4096

256 GB
-n 0x400000000000 -k 8192

512 GB
-n 0x1000000000000 -k 8192

1 TB
-n 0x1000000000000 -k 16384

2 TB
-n 0x4000000000000 -k 16384

4 TB
-n 0x4000000000000 -k 32768

8 TB
-n 0x10000000000000 -k 32768
