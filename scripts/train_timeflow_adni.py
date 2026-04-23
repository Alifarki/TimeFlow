import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / 'data'
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

import numpy as np
import pandas as pd
import torch
from mmengine import Config
from tqdm import tqdm

from models import TimeFlow, VecIntegrate, Warp, LocalNormalizedCrossCorrelationLoss, GradICONLoss, Fg_SDlogDetJac, FgPSNR
from adni_dataset_fixed import create_dataloaders
from utils import set_seed


def masked_l1(pred, target, mask):
    diff = (pred - target).abs() * mask
    return diff.sum() / mask.sum().clamp_min(1.0)


def masked_delta_l1(pred, gt, source, mask):
    return masked_l1(pred - source, gt - source, mask)


class DeltaPearsonCorrelation:
    def __call__(self, pred_image, gt_image, baseline_image, mask: Optional[torch.Tensor] = None) -> float:
        pred = pred_image.detach().cpu().numpy()
        gt = gt_image.detach().cpu().numpy()
        base = baseline_image.detach().cpu().numpy()
        dp = pred - base
        dg = gt - base
        if mask is not None:
            m = mask.detach().cpu().numpy() > 0
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
    return TimeFlow(t_embed_dim=args.t_embed_dim, encoder_cfg=encoder_cfg, decoder_cfg=decoder_cfg, remain_cfg=remain_cfg, pe_type='spe', max_periods=100).to(device)


def predict(model, warp, source, target, t, vecint=None):
    flow = model(source, target, t)
    if vecint is not None:
        flow = vecint(flow)
    return warp(source, flow), flow


def train_one_epoch(model, loader, optimizer, scaler, warp, vecint, loss_funcs, args, device):
    model.train()
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    running = {k: 0.0 for k in ['total','anchor_sim','future_sim','interp_sim','extrap_sim','interp_delta','extrap_delta','gradicon']}
    pbar = tqdm(loader, desc='Training', leave=False)
    for batch in pbar:
        source = batch['source'].float().to(device, non_blocking=True)
        middle = batch['middle'].float().to(device, non_blocking=True)
        future = batch['future'].float().to(device, non_blocking=True)
        t_interp = batch['t_interp'].float().to(device)
        t_extrap = batch['t_extrap'].float().to(device)
        fg = (source > 0).float()
        one = torch.ones((source.shape[0],), device=device, dtype=source.dtype)
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=args.use_amp):
            pred_anchor, flow_anchor = predict(model, warp, source, middle, one, vecint)
            pred_future, flow_future = predict(model, warp, source, future, one, vecint)
            pred_interp, _ = predict(model, warp, source, future, t_interp, vecint)
            pred_extrap, _ = predict(model, warp, source, middle, t_extrap, vecint)
            anchor_sim = loss_funcs['sim'](pred_anchor, middle)
            future_sim = loss_funcs['sim'](pred_future, future)
            interp_sim = loss_funcs['sim'](pred_interp, middle)
            extrap_sim = loss_funcs['sim'](pred_extrap, future)
            interp_delta = masked_delta_l1(pred_interp, middle, source, fg)
            extrap_delta = masked_delta_l1(pred_extrap, future, source, fg)
            total_loss = (
                args.anchor_sim_weight * anchor_sim +
                args.future_sim_weight * future_sim +
                args.interp_sim_weight * interp_sim +
                args.extrap_sim_weight * extrap_sim +
                args.interp_delta_weight * interp_delta +
                args.extrap_delta_weight * extrap_delta
            )
            gradicon = torch.tensor(0.0, device=device)
            if args.gradicon_weight > 0:
                flow_anchor_neg = model(source, middle, -one)
                flow_future_neg = model(source, future, -one)
                if vecint is not None:
                    flow_anchor_neg = vecint(flow_anchor_neg)
                    flow_future_neg = vecint(flow_future_neg)
                gradicon = 0.5 * (loss_funcs['gradicon'](flow_anchor, flow_anchor_neg) + loss_funcs['gradicon'](flow_future, flow_future_neg))
                total_loss = total_loss + args.gradicon_weight * gradicon
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        vals = [total_loss, anchor_sim, future_sim, interp_sim, extrap_sim, interp_delta, extrap_delta, gradicon]
        for k,v in zip(running.keys(), vals):
            running[k] += float(v.item())
        pbar.set_postfix(loss=f'{float(total_loss.item()):.4f}')
    n = max(len(loader), 1)
    return {k: v / n for k, v in running.items()}


@torch.no_grad()
def evaluate_triplets(model, loader, warp, vecint, metric_funcs, device):
    model.eval()
    rows = []
    for batch in tqdm(loader, desc='Evaluation', leave=False):
        source = batch['source'].float().to(device, non_blocking=True)
        middle = batch['middle'].float().to(device, non_blocking=True)
        future = batch['future'].float().to(device, non_blocking=True)
        t_interp = batch['t_interp'].float().to(device)
        t_extrap = batch['t_extrap'].float().to(device)
        fg = (source > 0).float()
        pred_interp, _ = predict(model, warp, source, future, t_interp, vecint)
        pred_extrap, _ = predict(model, warp, source, middle, t_extrap, vecint)
        flow_end = model(source, future, torch.ones((source.shape[0],), device=device, dtype=source.dtype))
        if vecint is not None:
            flow_end = vecint(flow_end)
        jac_sdlogj, jac_ndv = metric_funcs['jacdet'](flow_end.detach().cpu().numpy(), fg.detach().cpu().numpy())
        rows.append({
            'subject_id': batch['subject_id'][0] if isinstance(batch['subject_id'], list) else batch['subject_id'],
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
        return {k: 0.0 for k in ['interp_psnr','interp_delta_corr','extrap_psnr','extrap_delta_corr','all_psnr','all_delta_corr','sdlogj','ndv']}
    return {
        'interp_psnr': float(df['interp_psnr'].mean()),
        'interp_delta_corr': float(df['interp_delta_corr'].mean()),
        'extrap_psnr': float(df['extrap_psnr'].mean()),
        'extrap_delta_corr': float(df['extrap_delta_corr'].mean()),
        'all_psnr': float(pd.concat([df['interp_psnr'], df['extrap_psnr']]).mean()),
        'all_delta_corr': float(pd.concat([df['interp_delta_corr'], df['extrap_delta_corr']]).mean()),
        'sdlogj': float(df['sdlogj'].mean()),
        'ndv': float(df['ndv'].mean()),
    }


def main():
    p = argparse.ArgumentParser(description='Extrapolation-focused supervised TimeFlow training')
    p.add_argument('--data-root', type=str, required=True)
    p.add_argument('--voxel-size', type=str, default='1mm', choices=['1mm', '2mm', '4mm'])
    p.add_argument('--image-size', type=int, nargs=3, default=[128,128,128])
    p.add_argument('--min-interval-months', type=int, default=12)
    p.add_argument('--min-gap-months', type=int, default=6)
    p.add_argument('--max-extrap-t', type=float, default=2.5)
    p.add_argument('--t-embed-dim', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr-decay', type=float, default=0.999)
    p.add_argument('--anchor-sim-weight', type=float, default=0.5)
    p.add_argument('--future-sim-weight', type=float, default=0.5)
    p.add_argument('--interp-sim-weight', type=float, default=1.0)
    p.add_argument('--extrap-sim-weight', type=float, default=2.0)
    p.add_argument('--interp-delta-weight', type=float, default=1.0)
    p.add_argument('--extrap-delta-weight', type=float, default=2.0)
    p.add_argument('--gradicon-weight', type=float, default=0.05)
    p.add_argument('--save-dir', type=str, required=True)
    p.add_argument('--save-interval', type=int, default=10)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--use-amp', action='store_true')
    p.add_argument('--amp-dtype', type=str, default='bf16', choices=['bf16','fp16'])
    p.add_argument('--use-diffeomorphic', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    print('\n' + '=' * 70)
    print('LOADING TRIPLET DATASETS')
    print('=' * 70)
    train_loader, val_loader, test_loader, dataset_info = create_dataloaders(
        data_root=args.data_root,
        voxel_size=args.voxel_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_shape=tuple(args.image_size),
        min_interval_months=args.min_interval_months,
        min_gap_months=args.min_gap_months,
        max_extrap_t=args.max_extrap_t,
    )

    model = build_model(args, device)
    print(f'\nModel parameters: {sum(p.numel() for p in model.parameters()):,}')
    warp = Warp(image_size=tuple(args.image_size), interp_mode='bilinear').to(device)
    vecint = None
    if args.use_diffeomorphic:
        vecint = VecIntegrate(image_size=tuple(args.image_size), num_steps=7, interp_mode='bilinear').to(device)

    loss_funcs = {
        'sim': LocalNormalizedCrossCorrelationLoss(spatial_dims=3, kernel_size=9).to(device),
        'gradicon': GradICONLoss(flow_loss_cfg={'penalty': 'l2'}, image_size=tuple(args.image_size), interp_mode='bilinear', delta=1e-3).to(device),
    }
    metric_funcs = {'psnr': FgPSNR(max_val=1.0), 'jacdet': Fg_SDlogDetJac(), 'delta_corr': DeltaPearsonCorrelation()}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    best_val_extrap_delta = -1e9
    best_path = os.path.join(args.save_dir, 'best_model.pth')
    history_path = os.path.join(args.save_dir, 'training_history.csv')

    print('\n' + '=' * 70)
    print('STARTING TRAINING')
    print('=' * 70)
    with open(history_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','train_total','train_anchor_sim','train_future_sim','train_interp_sim','train_extrap_sim','train_interp_delta','train_extrap_delta','train_gradicon','val_interp_psnr','val_interp_delta_corr','val_extrap_psnr','val_extrap_delta_corr','val_all_psnr','val_all_delta_corr','val_sdlogj','val_ndv'])
        for epoch in range(args.epochs):
            print(f'\nEpoch {epoch+1}/{args.epochs}')
            print('-' * 70)
            train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, warp, vecint, loss_funcs, args, device)
            lr_scheduler.step()
            val_df = evaluate_triplets(model, val_loader, warp, vecint, metric_funcs, device)
            val_summary = summarize_triplet_df(val_df)
            print(f"Train - Total: {train_metrics['total']:.4f} | InterpΔ: {train_metrics['interp_delta']:.4f} | ExtrapΔ: {train_metrics['extrap_delta']:.4f}")
            print(f"Val   - Interp ΔCorr: {val_summary['interp_delta_corr']:.4f} | Extrap ΔCorr: {val_summary['extrap_delta_corr']:.4f} | All ΔCorr: {val_summary['all_delta_corr']:.4f} | Extrap PSNR: {val_summary['extrap_psnr']:.2f}")
            writer.writerow([epoch+1,train_metrics['total'],train_metrics['anchor_sim'],train_metrics['future_sim'],train_metrics['interp_sim'],train_metrics['extrap_sim'],train_metrics['interp_delta'],train_metrics['extrap_delta'],train_metrics['gradicon'],val_summary['interp_psnr'],val_summary['interp_delta_corr'],val_summary['extrap_psnr'],val_summary['extrap_delta_corr'],val_summary['all_psnr'],val_summary['all_delta_corr'],val_summary['sdlogj'],val_summary['ndv']])
            f.flush()
            if val_summary['extrap_delta_corr'] > best_val_extrap_delta:
                best_val_extrap_delta = val_summary['extrap_delta_corr']
                torch.save(model.state_dict(), best_path)
                val_df.to_csv(os.path.join(args.save_dir, 'best_val_triplet_metrics.csv'), index=False)
                print(f"  ✓ Saved best model (Extrapolation Delta-Corr: {best_val_extrap_delta:.4f})")
            if (epoch + 1) % args.save_interval == 0:
                torch.save(model.state_dict(), os.path.join(args.save_dir, f'checkpoint_epoch{epoch + 1:04d}.pth'))
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model.pth'))
    print('\n' + '=' * 70)
    print('FINAL TEST EVALUATION')
    print('=' * 70)
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    test_df = evaluate_triplets(model, test_loader, warp, vecint, metric_funcs, device)
    test_summary = summarize_triplet_df(test_df)
    test_df.to_csv(os.path.join(args.save_dir, 'test_triplet_results.csv'), index=False)
    print(f"Test Interpolation Delta-Corr: {test_summary['interp_delta_corr']:.4f}")
    print(f"Test Extrapolation Delta-Corr: {test_summary['extrap_delta_corr']:.4f}")
    print(f"Test All Delta-Corr:           {test_summary['all_delta_corr']:.4f}")
    print(f"Test Extrapolation PSNR:       {test_summary['extrap_psnr']:.2f}")
    print(f"Test Interpolation PSNR:       {test_summary['interp_psnr']:.2f}")
    print(f"\nTraining complete. Best validation extrapolation Delta-Corr: {best_val_extrap_delta:.4f}")
    print(f"Models and results saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
