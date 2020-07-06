# WBLC

```
Accepted paper @ IRC 2020, will soon be published.
M. Verucchi, L. Bartoli, F. Bagni, F. Gatti, P. Burgio and M. Bertogna, "Real-Time clustering and LiDAR-camera fusion on embedded platforms for self-driving cars",  in proceedings in IEEE Robotic Computing (2020)
```

## Dependencies

CUDA9.0+, Eigen and yaml. 

```
sudo apt-get install libeigen3-dev libyaml-cpp-dev
```

Tested on ubuntu 16.04 TLS with:

    - cmake 3.5.1
    - Eigen 3.3
    - yaml  0.5.2

## Testing

Alghoritm parameters setting in conf/config.yaml file.

Testing dataset contain in modenaDataset/

```
mkdir build
cd build
cmake ..
make
./WBLCtesting
```

## Testing on TX2

Nvidia TX2 need a different type of memory, because the pinned is not cached. The problem is descripted [here](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management).

You need to uncomment the define COMPILE_FOR_NVIDIA_TX2 collocated at line 12 in cudaNeighbours.h file.
