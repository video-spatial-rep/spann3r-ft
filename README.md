# spann3r-ft

## Log

    
| Date | News/TODOs |
|--|--|
| 20250206 | Llava-video encoder processing script & ckpt|
| 20250205 | update training script for vsi |

 ## Shortcuts

-  [data_preprocessing](https://github.com/video-spatial-rep/spann3r-ft/blob/main/docs/data_preprocess.md)
- [dust3r, spann3r, llava encoder](https://github.com/video-spatial-rep/spann3r-ft/blob/main/ckpt)

## Training [M3]
`CUDA_VISIBLE_DEVICES=0 python train.py --batch_size 8`
`python eval.py --exp_path /home/rilyn/project-files/02-pj-cambrians/spann3r/output/arkitscene_only_spann3r_1e-5/ --ckpt checkpoint-last.pth --device cuda:0`
