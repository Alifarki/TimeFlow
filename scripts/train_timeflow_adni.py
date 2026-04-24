import argparse
import csv
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / 'data'
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from mmengine import Config
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from models import TimeFlow, VecIntegrate, Warp, LocalNormalizedCrossCorrelationLoss, GradICONLoss, Fg_SDlogDetJac, FgPSNR
from adni_dataset_fixed import create_dataloaders
from utils import set_seed


# -----------------------------------------------------------------------------
# Distributed helpers, intentionally close to the working SADM DDP pattern.
# -----------------------------------------------------------------------------
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        args.distributed = args.world_size > 1
    elif 'SLURM_PROCID' in os.environ and int(os.environ.get('SLURM_NTASKS', 1)) > 1:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.local_rank = int(os.environ.get('SLURM_LOCALID', 0))
        args.distributed = True
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False

    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError('DDP was requested, but CUDA is not available.')
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=timedelta(minutes=args.ddp_timeout_minutes),
        )
        device = torch.device('cuda', args.local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def cleanup_distributed(args):
    if getattr(args, 'distributed', False) and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(args) -> bool:
    return int(getattr(args, 'rank', 0)) == 0


def barrier(args):
    if getattr(args, 'distributed', False) and dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def reduce_dict_mean(metrics: Dict[str, float], args, device: torch.device) -> Dict[str, float]:
    if not getattr(args, 'distributed', False):
        return metrics
    keys = sorted(metrics.keys())
    vals = torch.tensor([float(metrics[k]) for k in keys], device=device, dtype=torch.float64)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    vals = vals / dist.get_world_size()
    return {k: float(vals[i].item()) for i, k in enumerate(keys)}


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


# -----------------------------------------------------------------------------
# Losses and metrics.
# -----------------------------------------------------------------------------
def masked_l1(pred, target, mask):
    diff = (pred - target).abs() * mask
    return diff.sum() / mask.sum().clamp_min(1.0)


def masked_delta_l1(pred, gt, source, mask):
    return masked_l1(pred - source, gt - source, mask)


def masked_delta_corr_loss(pred, gt, source, mask, eps: float = 1e-8):
    """Differentiable loss = 1 - PearsonCorr(pred-source, gt-source) inside mask."""
    pd = (pred - source).float()
    gd = (gt - source).float()
    m = mask.float()
    if m.shape[0] == 1 and pd.shape[0] > 1:
        m = m.expand_as(pd)
    else:
        m = m.expand_as(pd)

    B = pd.shape[0]
    pd = pd.reshape(B, -1)
    gd = gd.reshape(B, -1)
    m = m.reshape(B, -1)

    denom_m = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    pd_mean = (pd * m).sum(dim=1, keepdim=True) / denom_m
    gd_mean = (gd * m).sum(dim=1, keepdim=True) / denom_m
    pd_c = (pd - pd_mean) * m
    gd_c = (gd - gd_mean) * m
    num = (pd_c * gd_c).sum(dim=1)
    den = torch.sqrt((pd_c.square().sum(dim=1) + eps) * (gd_c.square().sum(dim=1) + eps))
    corr = num / den.clamp_min(eps)
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return 1.0 - corr.mean()


class DeltaPearsonCorrelation:
    def __call__(self, pred_image, gt_image, baseline_image, mask: Optional[torch.Tensor] = None) -> float:
        pred = pred_image.detach().cpu().float().numpy()
        gt = gt_image.detach().cpu().float().numpy()
        base = baseline_image.detach().cpu().float().numpy()
        dp = pred - base
        dg = gt - base
        if mask is not None:
            m = mask.detach().cpu().numpy() > 0
            if m.shape[0] == 1 and dp.shape[0] > 1:
                m = np.broadcast_to(m, dp.shape)
            dp = dp[m]
            dg = dg[m]
        else:
            dp = dp.reshape(-1)
            dg = dg.reshape(-1)
        if dp.size < 2 or np.std(dp) < 1e-8 or np.std(dg) < 1e-8:
            return 0.0
        corr = np.corrcoef(dp, dg)[0, 1]
        return 0.0 if np.isnan(corr) else float(corr)


def mae_metric(pred, target, mask):
    diff = (pred - target).abs() * mask
    return float(diff.sum().item() / max(float(mask.sum().item()), 1.0))


def mse_metric(pred, target, mask):
    diff = ((pred - target) ** 2) * mask
    return float(diff.sum().item() / max(float(mask.sum().item()), 1.0))


# -----------------------------------------------------------------------------
# Model.
# -----------------------------------------------------------------------------
def build_model(args, device):
    encoder_cfg = Config({
        'spatial_dims': 3,
        'in_chan': 2,
        'down': True,
        'out_channels': [32, 32, 48, 48, 96],
        'out_indices': [0, 1, 2, 3, 4],
        'block_config': {
            'kernel_size': 3,
            't_embed_dim': args.t_embed_dim,
            'adaptive_norm': True,
            'down_first': True,
            'conv_down': True,
            'bias': True,
            'norm_name': ('INSTANCE', {'affine': False}),
            'act_name': ('leakyrelu', {'inplace': True, 'negative_slope': 0.2}),
            'dropout': None,
        },
    })
    decoder_cfg = Config({
        'spatial_dims': 3,
        'skip_channels': [96, 48, 48, 32, 32],
        'out_channels': [96, 48, 48, 32, 32],
        'block_config': {
            'kernel_size': 3,
            't_embed_dim': args.t_embed_dim,
            'adaptive_norm': True,
            'up_transp_conv': True,
            'transp_bias': False,
            'upsample_kernel_size': 2,
            'bias': True,
            'norm_name': ('INSTANCE', {'affine': False}),
            'act_name': ('leakyrelu', {'inplace': True, 'negative_slope': 0.2}),
            'dropout': None,
        },
    })
    remain_cfg = Config({
        'spatial_dims': 3,
        'in_chan': 32,
        'down': False,
        'out_channels': [32, 32],
        'out_indices': [0],
        'block_config': {
            'kernel_size': 3,
            't_embed_dim': args.t_embed_dim,
            'adaptive_norm': True,
            'bias': True,
            'norm_name': ('INSTANCE', {'affine': False}),
            'act_name': ('leakyrelu', {'inplace': True, 'negative_slope': 0.2}),
            'dropout': None,
        },
    })
    return TimeFlow(
        t_embed_dim=args.t_embed_dim,
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
        remain_cfg=remain_cfg,
        pe_type='spe',
        max_periods=100,
    ).to(device)


def predict(model, warp, source, target, t, vecint=None):
    flow = model(source, target, t)
    if vecint is not None:
        flow = vecint(flow)
    return warp(source, flow), flow


# -----------------------------------------------------------------------------
# Train / evaluate.
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, warp, vecint, loss_funcs, args, device):
    model.train()
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    autocast_enabled = bool(args.use_amp and device.type == 'cuda')
    running = {k: 0.0 for k in [
        'total', 'sim_total_raw', 'sim_total_scaled', 'anchor_sim', 'future_sim',
        'interp_sim', 'extrap_sim', 'interp_delta', 'extrap_delta',
        'interp_corr_loss', 'extrap_corr_loss', 'gradicon'
    ]}
    pbar = tqdm(loader, desc='Training', leave=False, disable=not is_main_process(args))

    for batch_idx, batch in enumerate(pbar):
        source = batch['source'].float().to(device, non_blocking=True)
        middle = batch['middle'].float().to(device, non_blocking=True)
        future = batch['future'].float().to(device, non_blocking=True)
        t_interp = batch['t_interp'].float().to(device)
        t_extrap = batch['t_extrap'].float().to(device)
        fg = (source > args.mask_threshold).float()
        one = torch.ones((source.shape[0],), device=device, dtype=source.dtype)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            pred_anchor, flow_anchor = predict(model, warp, source, middle, one, vecint)
            pred_future, flow_future = predict(model, warp, source, future, one, vecint)
            pred_interp, _ = predict(model, warp, source, future, t_interp, vecint)
            pred_extrap, _ = predict(model, warp, source, middle, t_extrap, vecint)

            anchor_sim = loss_funcs['sim'](pred_anchor, middle)
            future_sim = loss_funcs['sim'](pred_future, future)
            interp_sim = loss_funcs['sim'](pred_interp, middle)
            extrap_sim = loss_funcs['sim'](pred_extrap, future)
            sim_total_raw = (
                args.anchor_sim_weight * anchor_sim +
                args.future_sim_weight * future_sim +
                args.interp_sim_weight * interp_sim +
                args.extrap_sim_weight * extrap_sim
            )
            sim_total_scaled = args.sim_loss_scale * sim_total_raw

            interp_delta = masked_delta_l1(pred_interp, middle, source, fg)
            extrap_delta = masked_delta_l1(pred_extrap, future, source, fg)
            interp_corr_loss = masked_delta_corr_loss(pred_interp, middle, source, fg)
            extrap_corr_loss = masked_delta_corr_loss(pred_extrap, future, source, fg)

            total_loss = (
                sim_total_scaled +
                args.interp_delta_weight * interp_delta +
                args.extrap_delta_weight * extrap_delta +
                args.delta_corr_loss_weight * (interp_corr_loss + extrap_corr_loss)
            )

            gradicon = torch.tensor(0.0, device=device)
            if args.gradicon_weight > 0:
                flow_anchor_neg = model(source, middle, -one)
                flow_future_neg = model(source, future, -one)
                if vecint is not None:
                    flow_anchor_neg = vecint(flow_anchor_neg)
                    flow_future_neg = vecint(flow_future_neg)
                gradicon = 0.5 * (
                    loss_funcs['gradicon'](flow_anchor, flow_anchor_neg) +
                    loss_funcs['gradicon'](flow_future, flow_future_neg)
                )
                total_loss = total_loss + args.gradicon_weight * gradicon

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        vals = [
            total_loss, sim_total_raw, sim_total_scaled, anchor_sim, future_sim,
            interp_sim, extrap_sim, interp_delta, extrap_delta,
            interp_corr_loss, extrap_corr_loss, gradicon,
        ]
        for k, v in zip(running.keys(), vals):
            running[k] += float(v.detach().item())
        if is_main_process(args) and ((batch_idx + 1) % args.print_every == 0):
            pbar.set_postfix(loss=f'{float(total_loss.detach().item()):.4f}')

    n = max(len(loader), 1)
    return {k: v / n for k, v in running.items()}


@torch.no_grad()
def evaluate_triplets(model, loader, warp, vecint, metric_funcs, args, device, max_batches: int = 0):
    model.eval()
    rows = []
    pbar = tqdm(loader, desc='Evaluation', leave=False, disable=not is_main_process(args))
    for batch_idx, batch in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break
        source = batch['source'].float().to(device, non_blocking=True)
        middle = batch['middle'].float().to(device, non_blocking=True)
        future = batch['future'].float().to(device, non_blocking=True)
        t_interp = batch['t_interp'].float().to(device)
        t_extrap = batch['t_extrap'].float().to(device)
        fg = (source > args.mask_threshold).float()

        pred_interp, _ = predict(model, warp, source, future, t_interp, vecint)
        pred_extrap, _ = predict(model, warp, source, middle, t_extrap, vecint)
        flow_end = model(source, future, torch.ones((source.shape[0],), device=device, dtype=source.dtype))
        if vecint is not None:
            flow_end = vecint(flow_end)
        jac_sdlogj, jac_ndv = metric_funcs['jacdet'](flow_end.detach().cpu().numpy(), fg.detach().cpu().numpy())

        rows.append({
            'subject_id': batch['subject_id'][0] if isinstance(batch['subject_id'], list) else batch['subject_id'],
            'source_session': batch['source_session'][0] if isinstance(batch['source_session'], list) else batch['source_session'],
            'middle_session': batch['middle_session'][0] if isinstance(batch['middle_session'], list) else batch['middle_session'],
            'future_session': batch['future_session'][0] if isinstance(batch['future_session'], list) else batch['future_session'],
            't_interp': float(t_interp.detach().cpu().reshape(-1)[0]),
            't_extrap': float(t_extrap.detach().cpu().reshape(-1)[0]),
            'interp_mae': mae_metric(pred_interp, middle, fg),
            'interp_mse': mse_metric(pred_interp, middle, fg),
            'interp_psnr': float(metric_funcs['psnr'](pred_interp, middle, fg).mean().item()),
            'interp_delta_corr': float(metric_funcs['delta_corr'](pred_interp, middle, source, fg)),
            'extrap_mae': mae_metric(pred_extrap, future, fg),
            'extrap_mse': mse_metric(pred_extrap, future, fg),
            'extrap_psnr': float(metric_funcs['psnr'](pred_extrap, future, fg).mean().item()),
            'extrap_delta_corr': float(metric_funcs['delta_corr'](pred_extrap, future, source, fg)),
            'sdlogj': float(jac_sdlogj.mean()),
            'ndv': float(jac_ndv.mean()),
        })
    return pd.DataFrame(rows)


def summarize_triplet_df(df):
    if len(df) == 0:
        return {k: 0.0 for k in [
            'interp_psnr', 'interp_delta_corr', 'extrap_psnr', 'extrap_delta_corr',
            'all_psnr', 'all_delta_corr', 'sdlogj', 'ndv', 'mean_t_extrap'
        ]}
    return {
        'interp_psnr': float(df['interp_psnr'].mean()),
        'interp_delta_corr': float(df['interp_delta_corr'].mean()),
        'extrap_psnr': float(df['extrap_psnr'].mean()),
        'extrap_delta_corr': float(df['extrap_delta_corr'].mean()),
        'all_psnr': float(pd.concat([df['interp_psnr'], df['extrap_psnr']]).mean()),
        'all_delta_corr': float(pd.concat([df['interp_delta_corr'], df['extrap_delta_corr']]).mean()),
        'sdlogj': float(df['sdlogj'].mean()),
        'ndv': float(df['ndv'].mean()),
        'mean_t_extrap': float(df['t_extrap'].mean()),
    }


# -----------------------------------------------------------------------------
# Checkpointing.
# -----------------------------------------------------------------------------
def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith('module.') for k in state_dict.keys()):
        return state_dict
    return {k.replace('module.', '', 1): v for k, v in state_dict.items()}


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int, best_metric: float, args, dataset_info: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        'epoch': int(epoch),
        'model_state_dict': unwrap_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'best_val_extrap_delta': float(best_metric),
        'args': vars(args),
        'dataset_info': dataset_info,
    }
    torch.save(ckpt, path)


def save_model_weights(path: Path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(unwrap_model(model).state_dict(), path)


def find_resume_checkpoint(args) -> Optional[Path]:
    if args.resume:
        return Path(args.resume)
    if args.resume_epoch is not None:
        candidate = Path(args.save_dir) / f'checkpoint_epoch{args.resume_epoch:04d}.pth'
        if candidate.exists():
            return candidate
        last_candidate = Path(args.save_dir) / 'last_checkpoint.pth'
        if last_candidate.exists():
            return last_candidate
        raise FileNotFoundError(f'Could not find {candidate} or {last_candidate}')
    if args.auto_resume:
        candidate = Path(args.save_dir) / 'last_checkpoint.pth'
        if candidate.exists():
            return candidate
    return None


def load_checkpoint_if_needed(args, model, optimizer, scheduler, scaler, device):
    resume_path = find_resume_checkpoint(args)
    if resume_path is None:
        return 0, -1e9
    if not resume_path.exists():
        raise FileNotFoundError(f'Resume checkpoint does not exist: {resume_path}')

    if is_main_process(args):
        print(f'\n📂 Resuming from: {resume_path}', flush=True)
    ckpt = torch.load(resume_path, map_location=device)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(strip_module_prefix(ckpt['model_state_dict']), strict=True)
        if not args.resume_model_only:
            if ckpt.get('optimizer_state_dict') is not None:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                move_optimizer_state_to_device(optimizer, device)
            if ckpt.get('scheduler_state_dict') is not None and scheduler is not None:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if ckpt.get('scaler_state_dict') is not None and scaler is not None:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = int(ckpt.get('epoch', 0))
        best_val = float(ckpt.get('best_val_extrap_delta', -1e9))
    else:
        # Old model-only .pth.
        model.load_state_dict(strip_module_prefix(ckpt), strict=True)
        start_epoch = int(args.resume_start_epoch)
        best_val = -1e9

    if is_main_process(args):
        print(f'✓ Loaded checkpoint. Continuation starts at epoch {start_epoch + 1}.', flush=True)
    return start_epoch, best_val


# -----------------------------------------------------------------------------
# CLI and main.
# -----------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description='TimeFlow ADNI triplet training with DDP, resume, normalization choices, and delta-correlation loss',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data-root', type=str, required=True)
    p.add_argument('--voxel-size', type=str, default='1mm', choices=['1mm', '2mm', '4mm'])
    p.add_argument('--image-size', type=int, nargs=3, default=[128, 128, 128])
    p.add_argument('--normalization', type=str, default='percentile', choices=['percentile', 'none'])
    p.add_argument('--normalization-percentile', type=float, default=99.9)
    p.add_argument('--mask-threshold', type=float, default=0.0)

    p.add_argument('--min-interval-months', type=int, default=12)
    p.add_argument('--min-gap-months', type=int, default=6)
    p.add_argument('--max-extrap-t', type=float, default=2.5)
    p.add_argument('--t-embed-dim', type=int, default=16)

    p.add_argument('--batch-size', type=int, default=1, help='Per-GPU batch size. Global batch = batch-size * GPUs.')
    p.add_argument('--epochs', type=int, default=100, help='Total epochs, not additional epochs.')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr-decay', type=float, default=0.999)
    p.add_argument('--grad-clip', type=float, default=1.0)

    p.add_argument('--anchor-sim-weight', type=float, default=0.5)
    p.add_argument('--future-sim-weight', type=float, default=0.5)
    p.add_argument('--interp-sim-weight', type=float, default=1.0)
    p.add_argument('--extrap-sim-weight', type=float, default=2.0)
    p.add_argument('--sim-loss-scale', type=float, default=1e-5,
                   help='Your log showed LNCC around -50k. This scale prevents LNCC from drowning delta/correlation losses.')
    p.add_argument('--interp-delta-weight', type=float, default=1.0)
    p.add_argument('--extrap-delta-weight', type=float, default=2.0)
    p.add_argument('--delta-corr-loss-weight', type=float, default=0.5,
                   help='Differentiable 1-Pearson loss on predicted change maps.')
    p.add_argument('--gradicon-weight', type=float, default=0.05)

    p.add_argument('--save-dir', type=str, required=True)
    p.add_argument('--save-interval', type=int, default=10)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--val-every', type=int, default=1)
    p.add_argument('--max-val-batches', type=int, default=0, help='0 means evaluate all validation triplets.')
    p.add_argument('--print-every', type=int, default=25)

    p.add_argument('--use-amp', action='store_true')
    p.add_argument('--amp-dtype', type=str, default='bf16', choices=['bf16', 'fp16'])
    p.add_argument('--use-diffeomorphic', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--deterministic', action='store_true')

    p.add_argument('--resume', type=str, default='')
    p.add_argument('--resume-epoch', type=int, default=None)
    p.add_argument('--resume-start-epoch', type=int, default=0)
    p.add_argument('--resume-model-only', action='store_true')
    p.add_argument('--auto-resume', action='store_true')

    p.add_argument('--ddp-timeout-minutes', type=int, default=60)
    return p


def main():
    args = build_argparser().parse_args()
    device = init_distributed_mode(args)

    try:
        set_seed(args.seed + args.rank)
        torch.backends.cudnn.benchmark = not args.deterministic
        torch.backends.cudnn.deterministic = args.deterministic
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision('high')

        if is_main_process(args):
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        barrier(args)

        if is_main_process(args):
            print('\n' + '═' * 72)
            print('TIMEFLOW EXTRAPOLATION-FOCUSED TRAINING CONFIGURATION')
            print('═' * 72)
            print(f'Distributed:          {args.distributed}')
            print(f'World size / GPUs:    {args.world_size}')
            print(f'Rank / local rank:    {args.rank} / {args.local_rank}')
            print(f'Data Root:            {args.data_root}')
            print(f'Save Directory:       {args.save_dir}')
            print(f'Image Size:           {args.image_size}')
            print(f'Normalization:        {args.normalization}')
            print(f'Max Extrap t:         {args.max_extrap_t}')
            print(f'Per-GPU Batch Size:   {args.batch_size}')
            print(f'Global Batch Size:    {args.batch_size * args.world_size}')
            print(f'Epochs:               {args.epochs}')
            print(f'LR:                   {args.lr}')
            print(f'Sim loss scale:       {args.sim_loss_scale}')
            print(f'Delta corr loss w:    {args.delta_corr_loss_weight}')
            print(f'AMP:                  {args.use_amp} ({args.amp_dtype})')
            print('═' * 72, flush=True)

        train_loader, val_loader, test_loader, dataset_info = create_dataloaders(
            data_root=args.data_root,
            voxel_size=args.voxel_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_shape=tuple(args.image_size),
            min_interval_months=args.min_interval_months,
            min_gap_months=args.min_gap_months,
            max_extrap_t=args.max_extrap_t,
            normalization=args.normalization,
            percentile=args.normalization_percentile,
            distributed=args.distributed,
            rank=args.rank,
            world_size=args.world_size,
            seed=args.seed,
            verbose=is_main_process(args),
        )

        model = build_model(args, device)
        warp = Warp(image_size=tuple(args.image_size), interp_mode='bilinear').to(device)
        vecint = None
        if args.use_diffeomorphic:
            vecint = VecIntegrate(image_size=tuple(args.image_size), num_steps=7, interp_mode='bilinear').to(device)

        loss_funcs = {
            'sim': LocalNormalizedCrossCorrelationLoss(spatial_dims=3, kernel_size=9).to(device),
            'gradicon': GradICONLoss(
                flow_loss_cfg={'penalty': 'l2'},
                image_size=tuple(args.image_size),
                interp_mode='bilinear',
                delta=1e-3,
            ).to(device),
        }
        metric_funcs = {
            'psnr': FgPSNR(max_val=1.0),
            'jacdet': Fg_SDlogDetJac(),
            'delta_corr': DeltaPearsonCorrelation(),
        }

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
        # bf16 does not need scaling. fp16 does.
        scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and args.amp_dtype == 'fp16'))

        start_epoch, best_val_extrap_delta = load_checkpoint_if_needed(
            args, model, optimizer, scheduler, scaler, device
        )
        barrier(args)

        if args.distributed:
            model = DDP(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )

        if is_main_process(args):
            print(f'\nModel parameters: {sum(p.numel() for p in unwrap_model(model).parameters()):,}', flush=True)
            print('\n' + '=' * 72)
            print('STARTING TRAINING')
            print('=' * 72, flush=True)

        history_path = Path(args.save_dir) / 'training_history.csv'
        history_exists = history_path.exists()
        history_file = None
        writer = None
        if is_main_process(args):
            history_file = open(history_path, 'a' if history_exists and start_epoch > 0 else 'w', newline='')
            writer = csv.writer(history_file)
            if not history_exists or start_epoch == 0:
                writer.writerow([
                    'epoch', 'train_total', 'train_sim_total_raw', 'train_sim_total_scaled',
                    'train_anchor_sim', 'train_future_sim', 'train_interp_sim', 'train_extrap_sim',
                    'train_interp_delta', 'train_extrap_delta', 'train_interp_corr_loss',
                    'train_extrap_corr_loss', 'train_gradicon', 'val_interp_psnr',
                    'val_interp_delta_corr', 'val_extrap_psnr', 'val_extrap_delta_corr',
                    'val_all_psnr', 'val_all_delta_corr', 'val_sdlogj', 'val_ndv',
                    'val_mean_t_extrap', 'lr', 'world_size', 'normalization', 'max_extrap_t'
                ])

        last_val_summary = None
        for epoch in range(start_epoch, args.epochs):
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            if is_main_process(args):
                print(f'\nEpoch {epoch + 1}/{args.epochs}')
                print('-' * 72, flush=True)

            train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, warp, vecint, loss_funcs, args, device)
            train_metrics = reduce_dict_mean(train_metrics, args, device)
            scheduler.step()
            current_lr = float(optimizer.param_groups[0]['lr'])

            do_val = ((epoch + 1) % args.val_every == 0) or (epoch == 0) or (epoch + 1 == args.epochs)
            if is_main_process(args) and do_val:
                val_df = evaluate_triplets(
                    unwrap_model(model), val_loader, warp, vecint, metric_funcs, args, device,
                    max_batches=args.max_val_batches,
                )
                val_summary = summarize_triplet_df(val_df)
                last_val_summary = val_summary
            elif is_main_process(args):
                val_summary = last_val_summary or summarize_triplet_df(pd.DataFrame())
            else:
                val_summary = None

            if is_main_process(args):
                print(
                    f"Train - Total: {train_metrics['total']:.4f} | "
                    f"RawSim: {train_metrics['sim_total_raw']:.2f} | "
                    f"ScaledSim: {train_metrics['sim_total_scaled']:.4f} | "
                    f"InterpΔ: {train_metrics['interp_delta']:.4f} | "
                    f"ExtrapΔ: {train_metrics['extrap_delta']:.4f} | "
                    f"CorrLoss I/E: {train_metrics['interp_corr_loss']:.4f}/{train_metrics['extrap_corr_loss']:.4f} | "
                    f"LR: {current_lr:.3e}",
                    flush=True,
                )
                if do_val:
                    print(
                        f"Val   - Interp ΔCorr: {val_summary['interp_delta_corr']:.4f} | "
                        f"Extrap ΔCorr: {val_summary['extrap_delta_corr']:.4f} | "
                        f"All ΔCorr: {val_summary['all_delta_corr']:.4f} | "
                        f"Extrap PSNR: {val_summary['extrap_psnr']:.2f} | "
                        f"Mean t_extrap: {val_summary['mean_t_extrap']:.2f}",
                        flush=True,
                    )
                else:
                    print('Val   - skipped this epoch', flush=True)

                writer.writerow([
                    epoch + 1,
                    train_metrics['total'], train_metrics['sim_total_raw'], train_metrics['sim_total_scaled'],
                    train_metrics['anchor_sim'], train_metrics['future_sim'], train_metrics['interp_sim'],
                    train_metrics['extrap_sim'], train_metrics['interp_delta'], train_metrics['extrap_delta'],
                    train_metrics['interp_corr_loss'], train_metrics['extrap_corr_loss'], train_metrics['gradicon'],
                    val_summary['interp_psnr'], val_summary['interp_delta_corr'], val_summary['extrap_psnr'],
                    val_summary['extrap_delta_corr'], val_summary['all_psnr'], val_summary['all_delta_corr'],
                    val_summary['sdlogj'], val_summary['ndv'], val_summary['mean_t_extrap'], current_lr,
                    args.world_size, args.normalization, args.max_extrap_t,
                ])
                history_file.flush()

                save_checkpoint(
                    Path(args.save_dir) / 'last_checkpoint.pth',
                    model, optimizer, scheduler, scaler, epoch + 1, best_val_extrap_delta, args, dataset_info,
                )

                if do_val and val_summary['extrap_delta_corr'] > best_val_extrap_delta:
                    best_val_extrap_delta = val_summary['extrap_delta_corr']
                    save_model_weights(Path(args.save_dir) / 'best_model.pth', model)
                    save_checkpoint(
                        Path(args.save_dir) / 'best_checkpoint.pth',
                        model, optimizer, scheduler, scaler, epoch + 1, best_val_extrap_delta, args, dataset_info,
                    )
                    val_df.to_csv(Path(args.save_dir) / 'best_val_triplet_metrics.csv', index=False)
                    print(f"  ✓ Saved best model (Extrapolation Delta-Corr: {best_val_extrap_delta:.4f})", flush=True)

                if (epoch + 1) % args.save_interval == 0:
                    save_checkpoint(
                        Path(args.save_dir) / f'checkpoint_epoch{epoch + 1:04d}.pth',
                        model, optimizer, scheduler, scaler, epoch + 1, best_val_extrap_delta, args, dataset_info,
                    )
                    save_model_weights(Path(args.save_dir) / f'model_epoch{epoch + 1:04d}.pth', model)
                    print(f"  ✓ Saved checkpoint_epoch{epoch + 1:04d}.pth and model_epoch{epoch + 1:04d}.pth", flush=True)

            barrier(args)

        if is_main_process(args):
            save_model_weights(Path(args.save_dir) / 'final_model.pth', model)
            save_checkpoint(
                Path(args.save_dir) / 'final_checkpoint.pth',
                model, optimizer, scheduler, scaler, args.epochs, best_val_extrap_delta, args, dataset_info,
            )

            print('\n' + '=' * 72)
            print('FINAL TEST EVALUATION')
            print('=' * 72, flush=True)
            best_path = Path(args.save_dir) / 'best_model.pth'
            if best_path.exists():
                unwrap_model(model).load_state_dict(torch.load(best_path, map_location=device))
            test_df = evaluate_triplets(unwrap_model(model), test_loader, warp, vecint, metric_funcs, args, device)
            test_summary = summarize_triplet_df(test_df)
            test_df.to_csv(Path(args.save_dir) / 'test_triplet_results.csv', index=False)
            print(f"Test Interpolation Delta-Corr: {test_summary['interp_delta_corr']:.4f}")
            print(f"Test Extrapolation Delta-Corr: {test_summary['extrap_delta_corr']:.4f}")
            print(f"Test All Delta-Corr:           {test_summary['all_delta_corr']:.4f}")
            print(f"Test Extrapolation PSNR:       {test_summary['extrap_psnr']:.2f}")
            print(f"Test Interpolation PSNR:       {test_summary['interp_psnr']:.2f}")
            print(f"\nTraining complete. Best validation extrapolation Delta-Corr: {best_val_extrap_delta:.4f}")
            print(f"Models and results saved to: {args.save_dir}", flush=True)

        if history_file is not None:
            history_file.close()
        barrier(args)

    finally:
        cleanup_distributed(args)


if __name__ == '__main__':
    main()
