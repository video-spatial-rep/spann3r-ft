# Datasets



## Training datasets

### ScanNet++

1. Download the [dataset](https://kaldir.vc.in.tum.de/scannetpp/)
2. Pre-process the dataset using [pre-process code](https://github.com/Nik-V9/scannetpp) in SplaTAM to generate undistorted DSLR depth.
3. Place it in `./data/scannetpp`

NOTE: Scannetpp is a great dataset for debugging and test purposes.



### ScanNet

1. Download the [dataset](http://www.scan-net.org/)
2. Extract and organize the dataset using [pre-process script](https://github.com/nianticlabs/simplerecon/tree/main/data_scripts/scannet_wrangling_scripts) in SimpleRecon
3. Place it in `./data/scannet`



### ArkitScenes

1. Download the [dataset](https://github.com/apple/ARKitScenes/blob/9ec0b99c3cd55e29fc0724e1229e2e6c2909ab45/DATA.md)
2. Place it in `./data/arkit_lowres`

NOTE: Due to the limit of storage, we use low-resolution input to supervise Spann3R. Ideally, you can use a higher resolution i.e. `vga_wide`, as in DUSt3R, for training.



