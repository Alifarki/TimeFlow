import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class SubjectRecord:
    subject_id: str
    sessions: List[Dict[str, object]]


def extract_month(session_name: str) -> int:
    match = re.search(r"ses-M(\d+)", session_name)
    return int(match.group(1)) if match else 0


def find_nifti_file(session_dir: Path, voxel_size: str) -> Optional[Path]:
    if voxel_size == '1mm':
        nii_files = [f for f in session_dir.glob('mwp1*.nii') if not f.name.startswith('v4_s4_')]
    elif voxel_size == '2mm':
        nii_files = list(session_dir.glob('v2_*.nii'))
    elif voxel_size == '4mm':
        nii_files = list(session_dir.glob('v4_s4_*.nii'))
    else:
        nii_files = list(session_dir.glob('mwp1*.nii'))
    return nii_files[0] if nii_files else None


def preprocess_volume(path: str, target_shape: Tuple[int, int, int]) -> np.ndarray:
    img = nib.load(path).get_fdata().astype(np.float32)
    upper = np.percentile(img, 99.9)
    img = np.clip(img, 0.0, upper)
    if upper > 1e-8:
        img = img / upper
    img = np.clip(img, 0.0, 1.0)
    if img.shape != target_shape:
        zoom_factors = [t / s for t, s in zip(target_shape, img.shape)]
        img = scipy.ndimage.zoom(img, zoom=zoom_factors, order=1)
    return img.astype(np.float32)


class ADNITripletDataset(Dataset):
    def __init__(self, base_path: str, voxel_size: str = '1mm', target_shape: Tuple[int, int, int] = (128, 128, 128), min_interval_months: int = 12, min_gap_months: int = 6, max_extrap_t: float = 2.5, verbose: bool = True):
        self.base_path = Path(base_path)
        self.voxel_size = voxel_size
        self.target_shape = tuple(target_shape)
        self.min_interval_months = int(min_interval_months)
        self.min_gap_months = int(min_gap_months)
        self.max_extrap_t = float(max_extrap_t)
        self.verbose = verbose
        self.subjects = self._load_subjects()
        self.triplets = self._build_triplets()
        if self.verbose:
            total_scans = sum(len(s.sessions) for s in self.subjects)
            print(f"\n[ADNITripletDataset] Loaded from: {self.base_path}")
            print(f"  Voxel size: {self.voxel_size}")
            print(f"  Target shape: {self.target_shape}")
            print(f"  Min interval months: {self.min_interval_months}")
            print(f"  Min gap months: {self.min_gap_months}")
            print(f"  Max extrapolation t: {self.max_extrap_t}")
            print(f"  Total subjects: {len(self.subjects)}")
            print(f"  Total scans: {total_scans}")
            print(f"  Total triplets: {len(self.triplets)}")

    def _load_subjects(self) -> List[SubjectRecord]:
        subjects = []
        subject_dirs = sorted([d for d in self.base_path.iterdir() if d.is_dir() and d.name.startswith('sub-')])
        for subject_dir in subject_dirs:
            sessions = []
            for session_dir in sorted(subject_dir.iterdir()):
                if not session_dir.is_dir() or not session_dir.name.startswith('ses-'):
                    continue
                nii_path = find_nifti_file(session_dir, self.voxel_size)
                if nii_path is None:
                    continue
                month = extract_month(session_dir.name)
                sessions.append({'session': session_dir.name, 'month': month, 'path': str(nii_path)})
            if len(sessions) < 3:
                continue
            sessions.sort(key=lambda x: x['month'])
            total_interval = int(sessions[-1]['month']) - int(sessions[0]['month'])
            if total_interval < self.min_interval_months:
                continue
            subjects.append(SubjectRecord(subject_id=subject_dir.name, sessions=sessions))
        return subjects

    def _build_triplets(self) -> List[Dict[str, object]]:
        triplets = []
        for record in self.subjects:
            sessions = record.sessions
            n = len(sessions)
            for i in range(n - 2):
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n):
                        mi = int(sessions[i]['month'])
                        mj = int(sessions[j]['month'])
                        mk = int(sessions[k]['month'])
                        if mj - mi < self.min_gap_months:
                            continue
                        if mk - mj < self.min_gap_months:
                            continue
                        if mk - mi < self.min_interval_months:
                            continue
                        t_interp = (mj - mi) / (mk - mi)
                        t_extrap = (mk - mi) / (mj - mi)
                        if t_interp <= 0.0 or t_interp >= 1.0:
                            continue
                        if t_extrap <= 1.0 or t_extrap > self.max_extrap_t:
                            continue
                        triplets.append({'subject_id': record.subject_id, 'source': sessions[i], 'middle': sessions[j], 'future': sessions[k], 't_interp': float(t_interp), 't_extrap': float(t_extrap)})
        return triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.triplets[idx]
        source_np = preprocess_volume(item['source']['path'], self.target_shape)
        middle_np = preprocess_volume(item['middle']['path'], self.target_shape)
        future_np = preprocess_volume(item['future']['path'], self.target_shape)
        return {
            'subject_id': item['subject_id'],
            'source_session': item['source']['session'],
            'middle_session': item['middle']['session'],
            'future_session': item['future']['session'],
            'source_month': int(item['source']['month']),
            'middle_month': int(item['middle']['month']),
            'future_month': int(item['future']['month']),
            'source': torch.from_numpy(source_np[None, ...]),
            'middle': torch.from_numpy(middle_np[None, ...]),
            'future': torch.from_numpy(future_np[None, ...]),
            't_interp': torch.tensor(item['t_interp'], dtype=torch.float32),
            't_extrap': torch.tensor(item['t_extrap'], dtype=torch.float32),
        }


def create_dataloaders(data_root: str, voxel_size: str = '1mm', batch_size: int = 1, num_workers: int = 4, target_shape: Tuple[int, int, int] = (128,128,128), min_interval_months: int = 12, min_gap_months: int = 6, max_extrap_t: float = 2.5):
    data_root = Path(data_root)
    print('\n' + '=' * 70)
    print('LOADING ADNI TRIPLET DATASET FOR EXTRAPOLATION-FOCUSED TIMEFLOW')
    print('=' * 70)
    print(f'Data root: {data_root}')
    print(f'Voxel size: {voxel_size}')
    print(f'Normalization: Per-image percentile 99.9 -> [0,1]')
    print(f'Target shape: {target_shape}')
    print(f'Triplet batch size: {batch_size}')
    print(f'Min total interval: {min_interval_months} months')
    print(f'Min local gap: {min_gap_months} months')
    print(f'Max extrapolation t: {max_extrap_t}')
    print('=' * 70)
    train_dataset = ADNITripletDataset(str(data_root / 'train'), voxel_size, target_shape, min_interval_months, min_gap_months, max_extrap_t, True)
    val_dataset = ADNITripletDataset(str(data_root / 'val'), voxel_size, target_shape, min_interval_months, min_gap_months, max_extrap_t, True)
    test_dataset = ADNITripletDataset(str(data_root / 'test'), voxel_size, target_shape, min_interval_months, min_gap_months, max_extrap_t, True)
    loader_kwargs = dict(num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_kwargs)
    info = {'train_subjects': len(train_dataset.subjects), 'val_subjects': len(val_dataset.subjects), 'test_subjects': len(test_dataset.subjects), 'train_triplets': len(train_dataset), 'val_triplets': len(val_dataset), 'test_triplets': len(test_dataset)}
    print('\n' + '=' * 70)
    print('DATASET SUMMARY')
    print('=' * 70)
    print(f"Training subjects: {info['train_subjects']:,}")
    print(f"Validation subjects: {info['val_subjects']:,}")
    print(f"Test subjects: {info['test_subjects']:,}")
    print(f"Training triplets: {info['train_triplets']:,}")
    print(f"Validation triplets: {info['val_triplets']:,}")
    print(f"Test triplets: {info['test_triplets']:,}")
    print('=' * 70 + '\n')
    return train_loader, val_loader, test_loader, info
