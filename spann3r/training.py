import os
import sys
import math
import json
import time
import torch
import argparse
import datetime
import numpy as np
import torch.backends.cudnn as cudnn

import croco.utils.misc as misc 

from pathlib import Path
from typing import Sized
from shutil import copyfile
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from spann3r.model import Spann3R
from dust3r.losses import L21
from spann3r.datasets import *
from spann3r.loss import Regr3D_t, ConfLoss_t, Regr3D_t_ScaleShiftInv
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler
def get_args_parser():
    parser = argparse.ArgumentParser('Spann3R training', add_help=False)
    
    # Model
    parser.add_argument('--model', default="Spann3R(dus3r_name='/home/rilyn/project-files/02-pj-cambrians/cambrians-prep/LLaVA-NeXT/spann3r/spann3r.pth', use_feat=False, mem_pos_enc=False)",
                        type=str, help="string containing the model to build")
    parser.add_argument('--pretrained', default=None, help='path of a starting checkpoint')

    # Loss
    parser.add_argument('--train_criterion', 
                        default="ConfLoss_t(Regr3D_t(L21, norm_mode='avg_dis', fix_first=False), alpha=0.4)",
                        type=str, help="train criterion")
    
    # Train only on ArkitScene
    parser.add_argument('--train_dataset', 
                        default="ArkitScene(split='train', ROOT='/data_new/rilyn/raw/Training', resolution=224, transform=ColorJitter, max_thresh=100)", 
                        type=str, help="training set")

    # Remove evaluation (omit --test_dataset)

    # Training parameters
    parser.add_argument('--batch_size', default=2, type=int, help="Batch size per GPU")
    parser.add_argument('--epochs', default=120, type=int, help="Maximum number of epochs")
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-06, help='Lower LR bound')

    # Others
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')

    # Output
    parser.add_argument('--output_dir', default='./output/arkitscene_only', type=str, help="Path to save the output")
    
    return parser


# def get_args_parser():
#     parser = argparse.ArgumentParser('Spann3R training', add_help=False)
#     parser.add_argument('--model', default="Spann3R(dus3r_name='./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth', use_feat=False, mem_pos_enc=False)",
#                         type=str, help="string containing the model to build")
#     parser.add_argument('--pretrained', default=None, help='path of a starting checkpoint')
    
#     # Loss
#     parser.add_argument('--train_criterion', 
#                         default="ConfLoss_t(Regr3D_t(L21, norm_mode='avg_dis', fix_first=False), alpha=0.4)",
#                         type=str, help="train criterion")
#     parser.add_argument('--test_criterion', default="Regr3D_t_ScaleShiftInv(L21, gt_scale=True)")

#     # Datasets
#     parser.add_argument('--train_dataset', 
#                         default= "10000 @ Co3d(split='train', ROOT='./data/co3d_preprocessed_50', resolution=224, num_frames=5, mask_bg='rand', transform=ColorJitter) + 10000 @ Co3d(split='train', ROOT='./data/co3d_preprocessed_50', resolution=224, num_frames=5, mask_bg='rand', transform=ColorJitter, use_comb=False) + 10000 @ BlendMVS(split='train', ROOT='./data/blendmvg', resolution=224) + 10000 @ Scannetpp(split='train', ROOT='./data/scannetpp', resolution=224, transform=ColorJitter) + 10000 @ habitat(split='train', ROOT='./data/habitat_5frame', resolution=224, transform=ColorJitter) + 10000 @ Scannet(split='train', ROOT='./data/scannet', resolution=224, transform=ColorJitter, max_thresh=50) + 10000 @ ArkitScene(split='train', ROOT='./data/arkit_lowres', resolution=224, transform=ColorJitter, max_thresh=100)",
#                         required=False, type=str, help="training set")
#     parser.add_argument('--test_dataset', 
#                         default="Scannetpp(split='val', ROOT='./data/scannetpp', resolution=224, num_seq=1, kf_every=10, seed=777, full_video=True) + 1000 @ Co3d(split='test', ROOT='./data/co3d_preprocessed_50', resolution=224, num_frames=5, mask_bg=False, seed=777)", 
#                         type=str, help="testing set")
    
#      # Exp
#     parser.add_argument('--seed', default=0, type=int, help="Random seed")
    
#     # Training
#     parser.add_argument('--batch_size', default=2, type=int,
#                         help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
#     parser.add_argument('--batch_size_test', default=1, type=int,
#                         help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
#     parser.add_argument('--accum_iter', default=1, type=int,
#                         help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
#     parser.add_argument('--epochs', default=120, type=int, help="Maximum number of epochs for the scheduler")
    
#     parser.add_argument('--weight_decay', type=float, default=0.05, help="weight decay (default: 0.05)")
#     parser.add_argument('--lr', type=float, default=5e-5, metavar='LR', help='learning rate (absolute lr)')
#     parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
#                         help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
#     parser.add_argument('--min_lr', type=float, default=1e-06, metavar='LR',
#                         help='lower lr bound for cyclic schedulers that hit 0')
#     parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='epochs to warmup LR')

#     parser.add_argument('--amp', type=int, default=0,
#                         choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")
    
#     # others
#     parser.add_argument('--num_workers', default=2, type=int)
#     parser.add_argument('--num_workers_test', default=0, type=int)
#     parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
#     parser.add_argument('--local_rank', default=-1, type=int)
#     parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

#     parser.add_argument('--eval_freq', type=int, default=1, help='Test loss evaluation frequency')
#     parser.add_argument('--save_freq', default=1, type=int,
#                         help='frequence (number of epochs) to save checkpoint in checkpoint-last.pth')
#     parser.add_argument('--keep_freq', default=5, type=int,
#                         help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
#     parser.add_argument('--print_freq', default=20, type=int,
#                         help='frequence (number of iterations) to print infos while training')
    
#     parser.add_argument('--alpha_c2f', type=int, default=1, help='use alpha c2f')
    
#     # output dir 
#     parser.add_argument('--output_dir', default='./output/all_alpha04_lr05', type=str, help="path where to save the output")
    
#     return parser
    
@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, log_writer=None, prefix='test'):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)
    
    save_path = os.path.join(args.output_dir, f'eval_{epoch}')
        
    os.makedirs(save_path, exist_ok=True)

    for i, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        for view in batch:
            for name in 'img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres'.split():  # pseudo_focal
                if name not in view:
                    continue
                view[name] = view[name].to(device, non_blocking=True)
        
        
        preds, preds_all = model.forward(batch)
        
        if i < 100:
            images_all = []
            pts_all = []
            for j, view in enumerate(batch):
                img_idx = 0
                mask = view['depthmap'][img_idx:img_idx+1].cpu().numpy()!=0
                image = view['img'][img_idx:img_idx+1].permute(0, 2, 3, 1).cpu().numpy()[mask].reshape(-1, 3)
                
                pts = preds[j]['pts3d' if j==0 else 'pts3d_in_other_view'][img_idx:img_idx+1].detach().cpu().numpy()
                pts = pts[mask].reshape(-1, 3)
                
                images_all.append(image)
                pts_all.append(pts)
            images_all = np.concatenate(images_all, axis=0)
            


            pts_all = np.concatenate(pts_all, axis=0)
            # create open3d point cloud and save
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_all.reshape(-1, 3))
            pcd.colors = o3d.utility.Vector3dVector((images_all.reshape(-1, 3)+1.0)/2.0)
            o3d.io.write_point_cloud(os.path.join(save_path, view['dataset'][0]+f"_idx_{i}.ply"), pcd)
            
        
        loss, loss_details, loss_factor = criterion.compute_frame_loss(batch, preds_all)
        loss_value = float(loss)
        
        metric_logger.update(loss=float(loss_value), **loss_details)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix+'_'+name, val, 1000*epoch)

    return results
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args, log_writer=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    optimizer.zero_grad()
    
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        for view in batch:
            for name in 'img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres'.split():
                if name not in view:
                    continue
                view[name] = view[name].to(device, non_blocking=True)

        preds, preds_all = model.forward(batch)
        loss, loss_details, loss_factor = criterion.compute_frame_loss(batch, preds_all)
        loss += loss_factor
        loss_value = float(loss)

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= args.accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % args.accum_iter == 0, clip_grad=1.0)

        if (data_iter_step + 1) % args.accum_iter == 0:
            optimizer.zero_grad()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value, **loss_details)

        # Log training metrics to wandb
        wandb.log({
            "train_loss": loss_value,
            "learning_rate": lr,
            "epoch": epoch
        })

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
#                     data_loader: Sized, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler,
#                     args,
#                     log_writer=None):
#     assert torch.backends.cuda.matmul.allow_tf32 == True

#     model.train(True)
#     metric_logger = misc.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     accum_iter = args.accum_iter

#     if log_writer is not None:
#         print('log_dir: {}'.format(log_writer.log_dir))

#     if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
#         data_loader.dataset.set_epoch(epoch)
#     if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
#         data_loader.sampler.set_epoch(epoch)
        
#     epoch_ratio = epoch/args.epochs
#     if epoch_ratio < 0.75:
#         active_ratio = min(1, epoch/args.epochs*2.0)
#     else:
#         active_ratio = max(0.5, 1 - (epoch_ratio - 0.75) / 0.25)
#     data_loader.dataset.set_ratio(active_ratio)
#     #print(f"active thresh: {data_loader.datasets.dataset.active_thresh}")
    
    
#     optimizer.zero_grad()

#     for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
#         epoch_f = epoch + data_iter_step / len(data_loader)

#         # we use a per iteration (instead of per epoch) lr scheduler
#         if data_iter_step % accum_iter == 0:
#             misc.adjust_learning_rate(optimizer, epoch_f, args)
        
#         for view in batch:
#             for name in 'img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres'.split():  # pseudo_focal
#                 if name not in view:
#                     continue
#                 view[name] = view[name].to(device, non_blocking=True)
        
        
#         preds, preds_all = model.forward(batch)
#         loss, loss_details, loss_factor = criterion.compute_frame_loss(batch, preds_all)
#         loss += loss_factor     

#         loss_value = float(loss)

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value), force=True)
#             sys.exit(1)

#         loss /= accum_iter
#         norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
#                     update_grad=(data_iter_step + 1) % accum_iter == 0, clip_grad=1.0) # 
        
#         if (data_iter_step + 1) % accum_iter == 0:
#             optimizer.zero_grad()

#         del loss
#         del batch

#         lr = optimizer.param_groups[0]["lr"]
#         metric_logger.update(epoch=epoch_f)
#         metric_logger.update(lr=lr)
#         metric_logger.update(loss=loss_value, **loss_details)

#         if (data_iter_step + 1) % accum_iter == 0 and ((data_iter_step + 1) % (accum_iter * args.print_freq)) == 0:
#             loss_value_reduce = misc.all_reduce_mean(loss_value)  # MUST BE EXECUTED BY ALL NODES
#             if log_writer is None:
#                 continue
#             """ We use epoch_1000x as the x-axis in tensorboard.
#             This calibrates different curves when batch size changes.
#             """
#             epoch_1000x = int(epoch_f * 1000)
#             log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
#             log_writer.add_scalar('train_lr', lr, epoch_1000x)
#             log_writer.add_scalar('train_iter', epoch_1000x, epoch_1000x)
#             log_writer.add_scalar('active_ratio', active_ratio, epoch_1000x)
#             for name, val in loss_details.items():
#                 log_writer.add_scalar('train_'+name, val, epoch_1000x)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


import wandb

import wandb

def train(args):
    if misc.get_rank() == 0:  # Ensure wandb only runs on the main process
        wandb.init(
            project="spann3r-training",
            name=f"arkitscene_{args.epochs}epochs",
            config=vars(args),
        )

    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()

    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Auto resume from last checkpoint
    last_ckpt_fname = os.path.join(args.output_dir, 'checkpoint-last.pth')
    args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    # Set start_epoch
    args.start_epoch = 0 if args.resume is None else args.epochs  # Default to 0 if no resume

    print('Building train dataset: {}'.format(args.train_dataset))
    data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)

    print('Loading model: {}'.format(args.model))
    model = eval(args.model).to("cuda" if torch.cuda.is_available() else "cpu")

    train_criterion = eval(args.train_criterion).to("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.AdamW(
        misc.get_parameter_groups(model, args.weight_decay),
        lr=args.lr, betas=(0.9, 0.95)
    )
    loss_scaler = NativeScaler()

    log_writer = SummaryWriter(log_dir=args.output_dir) if global_rank == 0 else None

    print(f"Start training for {args.epochs} epochs")

    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, train_criterion, data_loader_train,
            optimizer, torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            epoch, loss_scaler, log_writer=log_writer, args=args
        )

        # Save model checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
            misc.save_model(args=args, model_without_ddp=model, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, fname="last", best_so_far=None)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    if global_rank == 0:
        wandb.log({"Total Training Time (s)": total_time})
        wandb.finish()

    print('Training time:', total_time_str)
