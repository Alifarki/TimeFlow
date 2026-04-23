import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / 'data'
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

import matplotlib.pyplot as plt
import pandas as pd
import torch
from mmengine import Config
from tqdm import tqdm

from models import FgPSNR, Fg_SDlogDetJac, TimeFlow, VecIntegrate, Warp
from adni_dataset_fixed import ADNITripletDataset


class DeltaPearsonCorrelation:
    def __call__(self, pred, gt, baseline, mask=None):
        pred = pred.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
        baseline = baseline.detach().cpu().numpy()
        dp = pred - baseline
        dg = gt - baseline
        if mask is not None:
            m = mask.detach().cpu().numpy() > 0
            dp = dp[m]
            dg = dg[m]
        else:
            dp = dp.reshape(-1)
            dg = dg.reshape(-1)
        if dp.size < 2 or dp.std() < 1e-8 or dg.std() < 1e-8:
            return 0.0
        c = __import__('numpy').corrcoef(dp, dg)[0, 1]
        return 0.0 if __import__('numpy').isnan(c) else float(c)


def build_model(args, device):
    encoder_cfg = Config({'spatial_dims': 3, 'in_chan': 2, 'down': True, 'out_channels': [32,32,48,48,96], 'out_indices': [0,1,2,3,4], 'block_config': {'kernel_size': 3, 't_embed_dim': args.t_embed_dim, 'adaptive_norm': True, 'down_first': True, 'conv_down': True, 'bias': True, 'norm_name': ('INSTANCE', {'affine': False}), 'act_name': ('leakyrelu', {'inplace': True, 'negative_slope': 0.2}), 'dropout': None}})
    decoder_cfg = Config({'spatial_dims': 3, 'skip_channels': [96,48,48,32,32], 'out_channels': [96,48,48,32,32], 'block_config': {'kernel_size': 3, 't_embed_dim': args.t_embed_dim, 'adaptive_norm': True, 'up_transp_conv': True, 'transp_bias': False, 'upsample_kernel_size': 2, 'bias': True, 'norm_name': ('INSTANCE', {'affine': False}), 'act_name': ('leakyrelu', {'inplace': True, 'negative_slope': 0.2}), 'dropout': None}})
    remain_cfg = Config({'spatial_dims': 3, 'in_chan': 32, 'down': False, 'out_channels': [32,32], 'out_indices': [0], 'block_config': {'kernel_size': 3, 't_embed_dim': args.t_embed_dim, 'adaptive_norm': True, 'bias': True, 'norm_name': ('INSTANCE', {'affine': False}), 'act_name': ('leakyrelu', {'inplace': True, 'negative_slope': 0.2}), 'dropout': None}})
    return TimeFlow(t_embed_dim=args.t_embed_dim, encoder_cfg=encoder_cfg, decoder_cfg=decoder_cfg, remain_cfg=remain_cfg, pe_type='spe', max_periods=100).to(device)


def mae_metric(pred, target, mask):
    diff = (pred - target).abs() * mask
    return float(diff.sum().item() / max(float(mask.sum().item()), 1.0))


def mse_metric(pred, target, mask):
    diff = ((pred - target) ** 2) * mask
    return float(diff.sum().item() / max(float(mask.sum().item()), 1.0))


def predict(model, warp, source, target, t, vecint=None):
    flow = model(source, target, t)
    if vecint is not None:
        flow = vecint(flow)
    return warp(source, flow), flow


@torch.no_grad()
def evaluate_split(model, dataset, warp, vecint, device):
    delta_metric = DeltaPearsonCorrelation()
    psnr_metric = FgPSNR(max_val=1.0)
    jac_metric = Fg_SDlogDetJac()
    rows = []
    for idx in tqdm(range(len(dataset)), desc=f"Triplets ({dataset.base_path.name})"):
        sample = dataset[idx]
        source = sample['source'][None].float().to(device, non_blocking=True)
        middle = sample['middle'][None].float().to(device, non_blocking=True)
        future = sample['future'][None].float().to(device, non_blocking=True)
        t_interp = sample['t_interp'][None].float().to(device)
        t_extrap = sample['t_extrap'][None].float().to(device)
        fg = (source > 0).float()
        pred_interp, _ = predict(model, warp, source, future, t_interp, vecint)
        pred_extrap, _ = predict(model, warp, source, middle, t_extrap, vecint)
        flow_end = model(source, future, torch.ones((1,), device=device, dtype=source.dtype))
        if vecint is not None:
            flow_end = vecint(flow_end)
        jac_sdlogj, jac_ndv = jac_metric(flow_end.detach().cpu().numpy(), fg.detach().cpu().numpy())
        rows.append({'subject_id': sample['subject_id'], 'source_session': sample['source_session'], 'middle_session': sample['middle_session'], 'future_session': sample['future_session'], 'source_month': sample['source_month'], 'middle_month': sample['middle_month'], 'future_month': sample['future_month'], 't_interp': float(sample['t_interp'].item()), 't_extrap': float(sample['t_extrap'].item()), 'interp_mae': mae_metric(pred_interp, middle, fg), 'interp_mse': mse_metric(pred_interp, middle, fg), 'interp_psnr': float(psnr_metric(pred_interp, middle, fg).mean().item()), 'interp_delta_corr': float(delta_metric(pred_interp, middle, source, fg)), 'extrap_mae': mae_metric(pred_extrap, future, fg), 'extrap_mse': mse_metric(pred_extrap, future, fg), 'extrap_psnr': float(psnr_metric(pred_extrap, future, fg).mean().item()), 'extrap_delta_corr': float(delta_metric(pred_extrap, future, source, fg)), 'sdlogj': float(jac_sdlogj.mean()), 'ndv': float(jac_ndv.mean())})
    return pd.DataFrame(rows)


def summarize_triplet_df(df, split_name):
    rows = []
    for setting, prefix in [('interpolation','interp'), ('extrapolation','extrap')]:
        rows.append({'split': split_name, 'setting': setting, 'num_samples': int(len(df)), 'mae_mean': float(df[f'{prefix}_mae'].mean()) if len(df) else 0.0, 'mae_std': float(df[f'{prefix}_mae'].std()) if len(df) else 0.0, 'mse_mean': float(df[f'{prefix}_mse'].mean()) if len(df) else 0.0, 'mse_std': float(df[f'{prefix}_mse'].std()) if len(df) else 0.0, 'psnr_mean': float(df[f'{prefix}_psnr'].mean()) if len(df) else 0.0, 'psnr_std': float(df[f'{prefix}_psnr'].std()) if len(df) else 0.0, 'delta_corr_mean': float(df[f'{prefix}_delta_corr'].mean()) if len(df) else 0.0, 'delta_corr_std': float(df[f'{prefix}_delta_corr'].std()) if len(df) else 0.0, 'sdlogj_mean': float(df['sdlogj'].mean()) if len(df) else 0.0, 'ndv_mean': float(df['ndv'].mean()) if len(df) else 0.0})
    rows.append({'split': split_name, 'setting': 'all', 'num_samples': int(2 * len(df)), 'mae_mean': float(pd.concat([df['interp_mae'], df['extrap_mae']]).mean()) if len(df) else 0.0, 'mae_std': float(pd.concat([df['interp_mae'], df['extrap_mae']]).std()) if len(df) else 0.0, 'mse_mean': float(pd.concat([df['interp_mse'], df['extrap_mse']]).mean()) if len(df) else 0.0, 'mse_std': float(pd.concat([df['interp_mse'], df['extrap_mse']]).std()) if len(df) else 0.0, 'psnr_mean': float(pd.concat([df['interp_psnr'], df['extrap_psnr']]).mean()) if len(df) else 0.0, 'psnr_std': float(pd.concat([df['interp_psnr'], df['extrap_psnr']]).std()) if len(df) else 0.0, 'delta_corr_mean': float(pd.concat([df['interp_delta_corr'], df['extrap_delta_corr']]).mean()) if len(df) else 0.0, 'delta_corr_std': float(pd.concat([df['interp_delta_corr'], df['extrap_delta_corr']]).std()) if len(df) else 0.0, 'sdlogj_mean': float(df['sdlogj'].mean()) if len(df) else 0.0, 'ndv_mean': float(df['ndv'].mean()) if len(df) else 0.0})
    return pd.DataFrame(rows)


def save_histograms(df, split_name, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    settings = [('Interpolation', df['interp_delta_corr'].dropna().values), ('Extrapolation', df['extrap_delta_corr'].dropna().values), ('All (Interpolation + Extrapolation)', pd.concat([df['interp_delta_corr'], df['extrap_delta_corr']]).dropna().values)]
    for ax, (title, values) in zip(axes, settings):
        if len(values):
            ax.hist(values, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {values.mean():.3f}')
            ax.legend()
        ax.set_title(f'{split_name}: {title}')
        ax.set_xlabel('Voxel-wise Delta Correlation')
        ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{split_name}_delta_correlation_histograms.png'), dpi=160, bbox_inches='tight')
    plt.close()


def main():
    p = argparse.ArgumentParser(description='Triplet-based evaluation for interpolation and extrapolation')
    p.add_argument('--data-root', type=str, required=True)
    p.add_argument('--splits', nargs='+', default=['val','test'], choices=['train','val','test'])
    p.add_argument('--model-path', type=str, required=True)
    p.add_argument('--save-dir', type=str, required=True)
    p.add_argument('--voxel-size', type=str, default='1mm', choices=['1mm','2mm','4mm'])
    p.add_argument('--image-size', type=int, nargs=3, default=[128,128,128])
    p.add_argument('--min-interval-months', type=int, default=12)
    p.add_argument('--min-gap-months', type=int, default=6)
    p.add_argument('--max-extrap-t', type=float, default=2.5)
    p.add_argument('--t-embed-dim', type=int, default=16)
    p.add_argument('--use-diffeomorphic', action='store_true')
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args, device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    warp = Warp(image_size=tuple(args.image_size), interp_mode='bilinear').to(device)
    vecint = VecIntegrate(image_size=tuple(args.image_size), num_steps=7, interp_mode='bilinear').to(device) if args.use_diffeomorphic else None

    all_details = []
    for split in args.splits:
        dataset = ADNITripletDataset(str(Path(args.data_root) / split), args.voxel_size, tuple(args.image_size), args.min_interval_months, args.min_gap_months, args.max_extrap_t, True)
        detail_df = evaluate_split(model, dataset, warp, vecint, device)
        detail_df.to_csv(os.path.join(args.save_dir, f'{split}_detailed_results.csv'), index=False)
        summary_df = summarize_triplet_df(detail_df, split)
        summary_df.to_csv(os.path.join(args.save_dir, f'{split}_summary.csv'), index=False)
        save_histograms(detail_df, split, args.save_dir)
        detail_df['split'] = split
        all_details.append(detail_df)
        print('\n' + '=' * 70)
        print(f'TIMEFLOW EVALUATION SUMMARY: {split}')
        print('=' * 70)
        for _, row in summary_df.iterrows():
            print(f"{row['setting'].upper():<15} N={int(row['num_samples'])} | MAE {row['mae_mean']:.4f} | MSE {row['mse_mean']:.6f} | PSNR {row['psnr_mean']:.2f} | DeltaCorr {row['delta_corr_mean']:.4f}")

    if all_details:
        combined_details = pd.concat(all_details, ignore_index=True)
        combined_details.to_csv(os.path.join(args.save_dir, 'combined_val_test_detailed_results.csv'), index=False)
        combined_summary = summarize_triplet_df(combined_details, 'combined')
        combined_summary.to_csv(os.path.join(args.save_dir, 'combined_val_test_summary.csv'), index=False)
        with open(os.path.join(args.save_dir, 'evaluation_metadata.json'), 'w') as f:
            json.dump({'model_path': args.model_path, 'splits': args.splits, 'voxel_size': args.voxel_size, 'image_size': list(args.image_size), 'min_interval_months': args.min_interval_months, 'min_gap_months': args.min_gap_months, 'max_extrap_t': args.max_extrap_t, 'use_diffeomorphic': args.use_diffeomorphic}, f, indent=2)
        print('\n' + '=' * 70)
        print('COMBINED SUMMARY')
        print('=' * 70)
        for _, row in combined_summary.iterrows():
            print(f"{row['setting'].upper():<15} N={int(row['num_samples'])} | MAE {row['mae_mean']:.4f} | MSE {row['mse_mean']:.6f} | PSNR {row['psnr_mean']:.2f} | DeltaCorr {row['delta_corr_mean']:.4f}")


if __name__ == '__main__':
    main()
