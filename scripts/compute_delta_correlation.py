#!/usr/bin/env python3
# scripts/compute_delta_correlation.py
# FIXED: Proper Config object usage for model building

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from scipy.ndimage import zoom
from tqdm import tqdm
import nibabel as nib
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from models import TimeFlow, Warp, VecIntegrate
from mmengine import Config


def build_timeflow_model():
    """Build TimeFlow model with proper Config objects"""
    t_embed_dim = 16
    adaptive_norm = True
    
    # FIXED: Use Config objects instead of dicts
    encoder_cfg = Config(dict(
        spatial_dims=3,
        in_chan=2,
        down=True,
        out_channels=[32, 32, 48, 48, 96],
        out_indices=[0, 1, 2, 3, 4],
        block_config=dict(
            kernel_size=3,
            t_embed_dim=t_embed_dim,
            adaptive_norm=adaptive_norm,
            down_first=True,
            conv_down=True,
            bias=True,
            norm_name=('INSTANCE', {'affine': False}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
            dropout=None,
        ),
    ))
    
    decoder_cfg = Config(dict(
        spatial_dims=3,
        skip_channels=[96, 48, 48, 32, 32],
        out_channels=[96, 48, 48, 32, 32],
        block_config=dict(
            kernel_size=3,
            t_embed_dim=t_embed_dim,
            adaptive_norm=adaptive_norm,
            up_transp_conv=True,
            transp_bias=False,
            upsample_kernel_size=2,
            bias=True,
            norm_name=('INSTANCE', {'affine': False}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
            dropout=None,
        ),
    ))
    
    remain_cfg = Config(dict(
        spatial_dims=3,
        in_chan=32,
        down=False,
        out_channels=[32, 32],
        out_indices=[1],
        block_config=dict(
            kernel_size=3,
            t_embed_dim=t_embed_dim,
            adaptive_norm=adaptive_norm,
            bias=True,
            norm_name=('INSTANCE', {'affine': False}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
            dropout=None,
        ),
    ))
    
    model = TimeFlow(
        t_embed_dim=t_embed_dim,
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
        remain_cfg=remain_cfg,
        pe_type='spe',
        max_periods=100,
    )
    
    return model


def compute_norm_stats_quick(train_dir: Path, target_shape: Tuple[int, int, int]) -> Dict:
    """Quickly compute normalization stats"""
    subject_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])[:10]
    
    all_mins, all_maxs = [], []
    
    for subject_dir in subject_dirs:
        session_dirs = sorted([d for d in subject_dir.iterdir() if d.is_dir()])
        if not session_dirs:
            continue
        
        nii_files = list(session_dirs[0].glob('mwp1*.nii'))
        if not nii_files:
            continue
        
        img = nib.load(str(nii_files[0])).get_fdata().astype(np.float32)
        all_mins.append(np.percentile(img, 0.5))
        all_maxs.append(np.percentile(img, 99.5))
    
    return {
        'p_min': np.mean(all_mins),
        'p_max': np.mean(all_maxs),
    }


def load_subject_sessions(subject_dir: Path, target_shape: Tuple, norm_stats: Dict, device: str) -> List[Dict]:
    """Load all sessions for a subject"""
    import re
    
    sessions = []
    
    for session_dir in sorted(subject_dir.iterdir()):
        if not session_dir.is_dir() or not session_dir.name.startswith('ses-'):
            continue
        
        nii_files = [f for f in session_dir.glob('mwp1*.nii') if not f.name.startswith('v4_s4_')]
        if not nii_files:
            continue
        
        match = re.search(r'ses-M(\d+)', session_dir.name)
        month = int(match.group(1)) if match else 0
        
        img = nib.load(str(nii_files[0])).get_fdata().astype(np.float32)
        
        if img.shape != target_shape:
            zoom_factors = [t / s for t, s in zip(target_shape, img.shape)]
            img = zoom(img, zoom_factors, order=1)
        
        p_min, p_max = norm_stats['p_min'], norm_stats['p_max']
        img = np.clip(img, p_min, p_max)
        if p_max - p_min > 1e-8:
            img = (img - p_min) / (p_max - p_min)
        
        img_tensor = torch.from_numpy(img[np.newaxis, np.newaxis, ...]).float().to(device)
        
        sessions.append({
            'month': month,
            'image': img_tensor,
            'path': str(nii_files[0]),
        })
    
    sessions.sort(key=lambda x: x['month'])
    return sessions


def compute_roi_correlation(pred_change: np.ndarray, actual_change: np.ndarray, mask: np.ndarray, n_rois: int = 27) -> float:
    """Compute ROI-based correlation"""
    grid_dim = int(np.round(n_rois ** (1/3)))
    
    H, W, D = pred_change.shape
    h_step = H // grid_dim
    w_step = W // grid_dim
    d_step = D // grid_dim
    
    roi_pred_means = []
    roi_actual_means = []
    
    for i in range(grid_dim):
        for j in range(grid_dim):
            for k in range(grid_dim):
                h_start, h_end = i * h_step, (i + 1) * h_step
                w_start, w_end = j * w_step, (j + 1) * w_step
                d_start, d_end = k * d_step, (k + 1) * d_step
                
                roi_mask = mask[h_start:h_end, w_start:w_end, d_start:d_end]
                
                if roi_mask.sum() < 10:
                    continue
                
                roi_pred = pred_change[h_start:h_end, w_start:w_end, d_start:d_end][roi_mask]
                roi_actual = actual_change[h_start:h_end, w_start:w_end, d_start:d_end][roi_mask]
                
                roi_pred_means.append(roi_pred.mean())
                roi_actual_means.append(roi_actual.mean())
    
    if len(roi_pred_means) < 3:
        return None
    
    corr, _ = pearsonr(roi_pred_means, roi_actual_means)
    return corr


def plot_delta_correlation_histogram(voxelwise_corrs: List[float], roi_corrs: List[float], output_file: Path):
    """Plot histogram of delta correlations"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if voxelwise_corrs:
        axes[0].hist(voxelwise_corrs, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(voxelwise_corrs), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(voxelwise_corrs):.3f}')
        axes[0].set_xlabel('Voxel-wise Delta Correlation')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Voxel-wise Delta Correlation (N={len(voxelwise_corrs)})')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    
    if roi_corrs:
        axes[1].hist(roi_corrs, bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[1].axvline(np.mean(roi_corrs), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(roi_corrs):.3f}')
        axes[1].set_xlabel('ROI-based Delta Correlation')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'ROI-based Delta Correlation (N={len(roi_corrs)})')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Histogram saved to: {output_file}")


def compute_delta_correlation_proper(
    model_path: str,
    data_root: str,
    split: str = 'test',
    device: str = 'cuda',
    output_dir: str = './delta_correlation_results',
    target_shape: Tuple[int, int, int] = (128, 128, 128),
    use_vecint: bool = False,
):
    """Compute Delta Pearson Correlation"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    model = build_timeflow_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    warp = Warp(image_size=target_shape, interp_mode='bilinear').to(device)
    
    if use_vecint:
        vecint = VecIntegrate(image_size=target_shape, num_steps=7, interp_mode='bilinear').to(device)
    else:
        vecint = None
    
    norm_stats = compute_norm_stats_quick(Path(data_root) / 'train', target_shape)
    
    test_dir = Path(data_root) / split
    subject_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    print(f"\nProcessing {len(subject_dirs)} subjects from {split} set...")
    
    all_voxelwise_correlations = []
    all_roi_correlations = []
    subject_results = []
    
    for subject_dir in tqdm(subject_dirs, desc="Computing delta correlations"):
        subject_id = subject_dir.name
        
        sessions = load_subject_sessions(subject_dir, target_shape, norm_stats, device)
        
        if len(sessions) < 4:
            continue
        
        baseline = sessions[0]
        
        for train_end_idx in range(2, len(sessions) - 1):
            train_endpoint = sessions[train_end_idx]
            
            baseline_month = baseline['month']
            train_endpoint_month = train_endpoint['month']
            interval = train_endpoint_month - baseline_month
            
            if interval < 12:
                continue
            
            for extrap_idx in range(train_end_idx + 1, len(sessions)):
                extrap_session = sessions[extrap_idx]
                extrap_month = extrap_session['month']
                
                t = (extrap_month - baseline_month) / interval
                
                with torch.no_grad():
                    t_tensor = torch.tensor([t], device=device, dtype=torch.float32)
                    flow = model(baseline['image'], train_endpoint['image'], t_tensor)
                    
                    if vecint:
                        flow = vecint(flow)
                    
                    predicted = warp(baseline['image'], flow)
                
                actual = extrap_session['image']
                
                baseline_np = baseline['image'].cpu().numpy().squeeze()
                predicted_np = predicted.cpu().numpy().squeeze()
                actual_np = actual.cpu().numpy().squeeze()
                
                predicted_change = predicted_np - baseline_np
                actual_change = actual_np - baseline_np
                
                mask = baseline_np > 0.01
                
                pred_change_flat = predicted_change[mask].flatten()
                actual_change_flat = actual_change[mask].flatten()
                
                if len(pred_change_flat) < 100:
                    continue
                
                corr, pval = pearsonr(pred_change_flat, actual_change_flat)
                all_voxelwise_correlations.append(corr)
                
                roi_corr = compute_roi_correlation(predicted_change, actual_change, mask, n_rois=27)
                
                if roi_corr is not None:
                    all_roi_correlations.append(roi_corr)
                
                subject_results.append({
                    'subject_id': subject_id,
                    'baseline_month': baseline_month,
                    'train_endpoint_month': train_endpoint_month,
                    'extrap_month': extrap_month,
                    't': t,
                    'voxelwise_correlation': corr,
                    'voxelwise_pvalue': pval,
                    'roi_correlation': roi_corr,
                })
    
    print(f"\n{'='*70}")
    print("DELTA CORRELATION RESULTS")
    print(f"{'='*70}")
    
    if all_voxelwise_correlations:
        voxelwise_mean = np.mean(all_voxelwise_correlations)
        voxelwise_std = np.std(all_voxelwise_correlations)
        voxelwise_median = np.median(all_voxelwise_correlations)
        
        print(f"\nVoxel-wise Delta Correlation:")
        print(f"  Mean:   {voxelwise_mean:.4f} ± {voxelwise_std:.4f}")
        print(f"  Median: {voxelwise_median:.4f}")
        print(f"  N samples: {len(all_voxelwise_correlations)}")
    
    if all_roi_correlations:
        roi_mean = np.mean(all_roi_correlations)
        roi_std = np.std(all_roi_correlations)
        roi_median = np.median(all_roi_correlations)
        
        print(f"\nROI-based Delta Correlation:")
        print(f"  Mean:   {roi_mean:.4f} ± {roi_std:.4f}")
        print(f"  Median: {roi_median:.4f}")
        print(f"  N samples: {len(all_roi_correlations)}")
    
    print(f"{'='*70}\n")
    
    results = {
        'voxelwise': {
            'correlations': all_voxelwise_correlations,
            'mean': float(np.mean(all_voxelwise_correlations)) if all_voxelwise_correlations else None,
            'std': float(np.std(all_voxelwise_correlations)) if all_voxelwise_correlations else None,
            'median': float(np.median(all_voxelwise_correlations)) if all_voxelwise_correlations else None,
            'n_samples': len(all_voxelwise_correlations),
        },
        'roi': {
            'correlations': all_roi_correlations,
            'mean': float(np.mean(all_roi_correlations)) if all_roi_correlations else None,
            'std': float(np.std(all_roi_correlations)) if all_roi_correlations else None,
            'median': float(np.median(all_roi_correlations)) if all_roi_correlations else None,
            'n_samples': len(all_roi_correlations),
        },
        'subject_results': subject_results,
    }
    
    results_file = output_path / f'{split}_delta_correlation.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    pickle_file = output_path / f'{split}_delta_correlation.pkl'
    with open(pickle_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Detailed results saved to: {pickle_file}")
    
    plot_delta_correlation_histogram(
        all_voxelwise_correlations,
        all_roi_correlations,
        output_path / f'{split}_delta_correlation_histogram.png'
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compute Delta Pearson Correlation for TimeFlow')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--output-dir', type=str, default='./delta_correlation_results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use-vecint', action='store_true')
    
    args = parser.parse_args()
    
    results = compute_delta_correlation_proper(
        model_path=args.model_path,
        data_root=args.data_root,
        split=args.split,
        device=args.device,
        output_dir=args.output_dir,
        use_vecint=args.use_vecint,
    )
    
    print("\n" + "="*70)
    print("DELTA CORRELATION COMPUTATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()