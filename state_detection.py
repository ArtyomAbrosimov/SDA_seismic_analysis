import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_selection import mutual_info_classif
import datetime

from SDA import SDA, StageMerging
from SDA.analytics import best_result

SAMPLE_SIZE = 30
FS = 100.0
EPOCH_DURATION = 0.8
OVERLAP = 0.5
STEP_DURATION = EPOCH_DURATION * (1 - OVERLAP)

SEISMIC_BANDS = {
    'ultra_low': (0.5, 2.0),
    'low': (2.0, 5.0),
    'mid': (5.0, 12.0),
    'high': (12.0, 25.0),
    'ultra_high': (25.0, 35.0)
}

BASE_PATH = "/Users/artyom/Desktop/Кактус/проекты/4 курс/Instance_sample_dataset_v3"
DATA_PATH = os.path.join(BASE_PATH, "data", "Instance_events_gm_10k.hdf5")
EVENTS_METADATA_PATH = os.path.join(BASE_PATH, "metadata", "metadata_Instance_events_10k.csv")

BASE_OUTPUT_DIR = "./runs"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
RANDOM_SEED = 42


class Loader:
    def __init__(self, data_path, metadata_path, sample_size=None, random_seed=RANDOM_SEED):
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.sample_size = sample_size
        self.random_seed = random_seed

    def load_data(self):
        metadata_df = pd.read_csv(self.metadata_path)
        waveforms = []
        metadata_list = []

        with h5py.File(self.data_path, 'r') as f:
            data_group = f['data']
            np.random.seed(self.random_seed)
            available_indices = list(range(len(metadata_df)))
            load_limit = self.sample_size if self.sample_size else len(metadata_df)

            if load_limit < len(available_indices):
                selected_indices = np.random.choice(available_indices, size=load_limit, replace=False)
            else:
                selected_indices = available_indices

            for idx in selected_indices:
                row = metadata_df.iloc[idx]
                trace_name = row['trace_name']
                waveform = data_group[trace_name][:]
                waveform = signal.detrend(waveform, axis=1)
                sos = signal.butter(4, [0.5, 35.0], btype='bandpass', fs=FS, output='sos')
                waveform = signal.sosfilt(sos, waveform, axis=1)
                waveform = waveform / np.max(np.abs(waveform))
                waveforms.append(waveform)
                metadata_list.append(row)

        return waveforms, pd.DataFrame(metadata_list)


class FeatureExtractor:
    def __init__(self, fs, epoch_len, overlap=OVERLAP):
        self.fs = fs
        self.epoch_samples = int(fs * epoch_len)
        self.step = int(self.epoch_samples * (1 - overlap))
        self.feature_names = []

    def _compute_polarization(self, epoch_data):
        cov = np.cov(epoch_data)
        vals, vecs = np.linalg.eigh(cov)
        idx = vals.argsort()[::-1]
        l1, l2, l3 = vals[idx]
        rectilinearity = 1.0 - ((l2 + l3) / (2 * l1)) if l1 > 0 else 0
        planarity = 1.0 - (2 * l3 / (l1 + l2)) if (l1 + l2) > 0 else 0
        log_energy = np.log1p(l1 + l2 + l3)
        return [rectilinearity, planarity, log_energy]

    def _compute_spectral_features(self, sig, channel_name=''):
        freqs, psd = signal.welch(sig, fs=self.fs, nperseg=min(256, len(sig)))
        total_power = np.trapz(psd, freqs) + 1e-12
        features = dict()

        features[f'{channel_name}_log_energy'] = np.log1p(np.sum(sig ** 2))
        features[f'{channel_name}_rms'] = np.sqrt(np.mean(sig ** 2))
        features[f'{channel_name}_kurtosis'] = stats.kurtosis(sig)
        features[f'{channel_name}_centroid'] = np.sum(freqs * psd) / total_power

        half = len(sig) // 2
        sta = np.mean(np.abs(sig[half:]))
        lta = np.mean(np.abs(sig[:half])) + 1e-6
        features[f'{channel_name}_sta_lta'] = np.log1p(sta / lta)

        for band_name, (f_low, f_high) in SEISMIC_BANDS.items():
            mask = (freqs >= f_low) & (freqs <= f_high)

            if np.any(mask):
                band_energy = np.trapz(psd[mask], freqs[mask])
                features[f'{channel_name}_energy_{band_name}'] = np.log1p(band_energy)
                features[f'{channel_name}_rel_power_{band_name}'] = band_energy / total_power
            else:
                features[f'{channel_name}_energy_{band_name}'] = 0
                features[f'{channel_name}_rel_power_{band_name}'] = 0

        if 'energy_high' in features and 'energy_low' in features:
            features[f'{channel_name}_S_over_P'] = np.log1p(
                features[f'{channel_name}_energy_high'] /
                (features[f'{channel_name}_energy_low'] + 1e-6)
            )

        return features

    def extract(self, waveform):
        n_channels, n_samples = waveform.shape
        features_list = []

        for start in range(0, n_samples - self.epoch_samples + 1, self.step):
            end = start + self.epoch_samples
            epoch = waveform[:, start:end]
            epoch_features = []
            epoch_features.extend(self._compute_polarization(epoch))

            for ch_idx, ch_name in enumerate(['Z', 'N', 'E']):
                ch_features = self._compute_spectral_features(epoch[ch_idx], ch_name)

                key_features = [
                    f'{ch_name}_log_energy',
                    f'{ch_name}_rms',
                    f'{ch_name}_kurtosis',
                    f'{ch_name}_centroid',
                    f'{ch_name}_sta_lta',
                    f'{ch_name}_S_over_P'
                ]

                for key in key_features:
                    epoch_features.append(ch_features.get(key, 0))

                for band_name in SEISMIC_BANDS.keys():
                    energy_key = f'{ch_name}_energy_{band_name}'
                    rel_power_key = f'{ch_name}_rel_power_{band_name}'
                    epoch_features.append(ch_features.get(energy_key, 0))
                    epoch_features.append(ch_features.get(rel_power_key, 0))

            features_list.append(epoch_features)

        if not self.feature_names:
            self.feature_names = ['rectilinearity', 'planarity', 'log_energy_polar']

            for ch_name in ['Z', 'N', 'E']:
                self.feature_names.extend([
                    f'{ch_name}_log_energy',
                    f'{ch_name}_rms',
                    f'{ch_name}_kurtosis',
                    f'{ch_name}_centroid',
                    f'{ch_name}_sta_lta',
                    f'{ch_name}_S_over_P'
                ])

                for band_name in SEISMIC_BANDS.keys():
                    self.feature_names.append(f'{ch_name}_energy_{band_name}')
                    self.feature_names.append(f'{ch_name}_rel_power_{band_name}')

        return np.array(features_list), self.feature_names


def calculate_metrics_for_states(features, st_edges):
    n_stages = len(st_edges) - 1

    if n_stages < 2:
        return {
            'avg_silhouette': 0,
            'avg_calinski_harabasz': 0,
            'avg_davies_bouldin': 0,
            'avg_ward_distance': 0
        }

    silhouette_scores = []
    ch_scores = []
    db_scores = []
    ward_distances = []

    for i in range(n_stages - 1):
        stage1 = features[st_edges[i]:st_edges[i + 1]]
        stage2 = features[st_edges[i + 1]:st_edges[i + 2]]

        if len(stage1) > 1 and len(stage2) > 1:
            pair_data = np.vstack([stage1, stage2])
            pair_labels = np.array([0] * len(stage1) + [1] * len(stage2))

            sil_score = silhouette_score(pair_data, pair_labels)
            silhouette_scores.append(sil_score)

            ch_scores.append(calinski_harabasz_score(pair_data, pair_labels))
            db_scores.append(davies_bouldin_score(pair_data, pair_labels))

            centroid1 = np.mean(stage1, axis=0)
            centroid2 = np.mean(stage2, axis=0)
            combined_centroid = np.mean(pair_data, axis=0)

            ward_dist = (
                    np.sum(np.linalg.norm(pair_data - combined_centroid, axis=1) ** 2) -
                    np.sum(np.linalg.norm(stage1 - centroid1, axis=1) ** 2) -
                    np.sum(np.linalg.norm(stage2 - centroid2, axis=1) ** 2)
            )
            ward_distances.append(ward_dist)

    return {
        'avg_silhouette': np.mean(silhouette_scores) if silhouette_scores else 0,
        'avg_calinski_harabasz': np.mean(ch_scores) if ch_scores else 0,
        'avg_davies_bouldin': np.mean(db_scores) if db_scores else 0,
        'avg_ward_distance': np.mean(ward_distances) if ward_distances else 0
    }


def calculate_IV(features, labels, feature_names, n_bins=10):
    n_classes = len(np.unique(labels))
    results = []
    mi_values = mutual_info_classif(features, labels, random_state=RANDOM_SEED)

    for f_idx, f_name in enumerate(feature_names):
        f_val = features[:, f_idx]
        bins = np.percentile(f_val, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)

        if len(bins) < 2:
            digitized = np.zeros_like(f_val, dtype=int)
        else:
            digitized = np.digitize(f_val, bins) - 1

        class_ivs = []

        for cls in range(n_classes):
            target = (labels == cls).astype(int)
            iv_cls = 0
            unique_bins = np.unique(digitized)
            p_good_global = np.mean(target)
            p_bad_global = 1 - p_good_global

            for b_val in unique_bins:
                mask = digitized == b_val
                if np.sum(mask) == 0:
                    continue

                p_good = np.sum(target[mask]) / np.sum(mask)
                p_bad = 1 - p_good

                if p_good > 0 and p_bad > 0 and p_good_global > 0:
                    woe = np.log((p_good / p_bad) / (p_good_global / p_bad_global))
                    iv_cls += (p_good - p_bad) * woe

            class_ivs.append(iv_cls)

        results.append({
            'Feature': f_name,
            'IV': np.mean(class_ivs) if class_ivs else 0,
            'Mutual_Info': mi_values[f_idx]
        })

    return pd.DataFrame(results)


def analyze_IV(features, labels, feature_names):
    df_IV = calculate_IV(features, labels, feature_names)
    iv_vals = df_IV['IV'].values
    iv_results = {
        'mean_IV': float(np.mean(iv_vals)),
        'n_features_IV_ge_0.4': int((iv_vals >= 0.4).sum()),
        'pct_features_IV_ge_0.4': float((iv_vals >= 0.4).sum() / len(iv_vals) * 100)
    }
    return df_IV, iv_results


def run_sda_alg(features):
    scaler = RobustScaler(quantile_range=(25, 75))
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=0.85, whiten=True)
    features_pca = pca.fit_transform(features_scaled)

    sda = SDA(
        n_jobs=4, scale=False, verbose=False, random_state=RANDOM_SEED,
        n_clusters_min=2, n_clusters_max=10,
        k_neighbours_min=3, k_neighbours_max=15,
        st1_merging=StageMerging.BOTH,
        st1_len_thresholds=[3, 5],
        st1_dist_rate=0.3,
        st2_merging=StageMerging.BOTH,
        st2_len_thresholds=[5],
        st2_dist_rate=0.25
    )

    results_df, df_st_edges = sda.apply(features_pca)
    best = best_result(results_df, 'Ward_dist', n_stages=None, min_stage_length=5)
    return best, features_pca, results_df


def plot_seismogram_with_stages(waveform, metadata, st_edges, seismogram_idx, step_duration, title_suffix=""):
    n_samples = waveform.shape[1]
    time = np.arange(n_samples) / FS
    fig, axes = plt.subplots(4, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [3, 3, 3, 1]})
    channel_names = ['Z (Vertical)', 'N (North)', 'E (East)']
    n_stages = len(st_edges) - 1
    colors = plt.cm.Set3(np.linspace(0, 1, max(n_stages, 1)))

    for ch_idx in range(3):
        ax = axes[ch_idx]
        ax.plot(time, waveform[ch_idx], 'k-', linewidth=0.5, alpha=0.7)

        for i, (start, end) in enumerate(zip(st_edges[:-1], st_edges[1:])):
            t_start = start * step_duration
            t_end = end * step_duration
            ax.axvspan(t_start, t_end, alpha=0.2, color=colors[i % len(colors)])

            if 'trace_P_arrival_sample' in metadata and not np.isnan(metadata['trace_P_arrival_sample']):
                ax.axvline(metadata['trace_P_arrival_sample'] / FS, color='r', ls='--', alpha=0.5,
                           label='P-wave' if ch_idx == 0 else '')
            if 'trace_S_arrival_sample' in metadata and not np.isnan(metadata['trace_S_arrival_sample']):
                ax.axvline(metadata['trace_S_arrival_sample'] / FS, color='b', ls='--', alpha=0.5,
                           label='S-wave' if ch_idx == 0 else '')

        ax.set_ylabel(channel_names[ch_idx])
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, time[-1])
        if ch_idx == 0:
            ax.legend(loc='upper right')

    ax_stage = axes[3]
    for i, (start, end) in enumerate(zip(st_edges[:-1], st_edges[1:])):
        t_start = start * step_duration
        t_end = end * step_duration
        width = t_end - t_start
        ax_stage.barh(0, width, left=t_start, height=0.5,
                      color=colors[i % len(colors)], edgecolor='black', alpha=0.7)

        if width > 2.0:
            label = f'State{i + 1}'
            ax_stage.text((t_start + t_end) / 2, 0, label, ha='center', va='center', fontweight='bold')

    ax_stage.set_yticks([])
    ax_stage.set_xlabel('Время (с)', fontsize=11)
    ax_stage.set_xlim(0, time[-1])
    plt.suptitle(f"Сейсмограмма {metadata.get('trace_name', seismogram_idx)} {title_suffix}", fontsize=14)
    plt.tight_layout()
    return fig


def plot_physical_features_analysis(features, feature_names, stage_labels, seismogram_idx):
    physical_keywords = ['rectilinearity', 'planarity', 'log_energy', 'kurtosis', 'rms', 'energy_', 'centroid',
                         'S_over_P', 'sta_lta']
    indices = []

    for i, name in enumerate(feature_names):
        if any(keyword in name for keyword in physical_keywords):
            indices.append(i)

    if len(indices) > 9:
        variances = np.var(features[:, indices], axis=0)
        top_indices = np.argsort(variances)[-9:][::-1]
        indices = [indices[i] for i in top_indices]

    n_feats = len(indices)
    if n_feats == 0:
        indices = list(range(min(6, features.shape[1])))
        n_feats = len(indices)

    rows = int(np.ceil(np.sqrt(n_feats)))
    cols = int(np.ceil(n_feats / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    if n_feats == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    unique_stages = np.unique(stage_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_stages)))

    for i, (feat_idx, ax) in enumerate(zip(indices, axes)):
        data = []
        for st in unique_stages:
            data.append(features[stage_labels == st, feat_idx])

        bplot = ax.boxplot(data, patch_artist=True, labels=[f'State{st + 1}' for st in unique_stages])
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(feature_names[feat_idx], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticklabels([f'S{st + 1}' for st in unique_stages], rotation=45)

    for i in range(n_feats, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f"Распределение признаков по стадиям - {seismogram_idx}", fontsize=14)
    plt.tight_layout()
    return fig


def plot_clustering_metrics_comparison(clustering_metrics, surrogate_metrics, seismogram_idx):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [('avg_silhouette', 'Silhouette'), ('avg_calinski_harabasz', 'CH Index'),
               ('avg_ward_distance', 'Ward Dist')]

    for i, (key, name) in enumerate(metrics):
        val_real = clustering_metrics.get(key, 0)
        val_surr = surrogate_metrics.get(key, 0)
        axes[i].bar(['Real', 'Surrogate'], [val_real, val_surr], color=['steelblue', 'lightcoral'])
        axes[i].set_title(name)
        axes[i].text(0, val_real, f"{val_real:.3f}", ha='center', va='bottom')
        axes[i].text(1, val_surr, f"{val_surr:.3f}", ha='center', va='bottom')
        axes[i].set_ylim([0, max(val_real, val_surr) * 1.2])

    plt.suptitle(f"Метрики кластеризации: Реальные vs Суррогат ({seismogram_idx})")
    plt.tight_layout()
    return fig


def plot_IV_analysis(df_IV_real, df_IV_surrogate, seismogram_idx):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]

    if df_IV_real is not None and not df_IV_real.empty:
        ax.hist(df_IV_real['IV'], alpha=0.5, label='Real', density=True, bins=15)
    if df_IV_surrogate is not None and not df_IV_surrogate.empty:
        ax.hist(df_IV_surrogate['IV'], alpha=0.5, label='Surrogate', density=True, bins=15)

    ax.legend()
    ax.set_title("Распределение IV ")
    ax.set_xlabel("IV ")
    ax.set_ylabel("Density ")
    ax = axes[1]

    if df_IV_real is not None and not df_IV_real.empty:
        top = df_IV_real.nlargest(10, 'IV').sort_values('IV')
        ax.barh(top['Feature'], top['IV'], color='steelblue')
        ax.set_title("Топ-10 признаков (Real) ")
        ax.set_xlabel("IV ")

    plt.suptitle(f"IV Анализ - {seismogram_idx} ")
    plt.tight_layout()
    return fig


def plot_feature_importance_heatmap(features, feature_names, stage_labels, seismogram_idx):
    unique_stages = np.unique(stage_labels)
    n_stages = len(unique_stages)
    variances = np.var(features, axis=0)
    top_indices = np.argsort(variances)[-20:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    mean_matrix = np.zeros((len(top_features), n_stages))

    for i, feat_idx in enumerate(top_indices):
        for j, stage in enumerate(unique_stages):
            stage_data = features[stage_labels == stage, feat_idx]
            mean_matrix[i, j] = np.mean(stage_data)

    row_max = mean_matrix.max(axis=1, keepdims=True)
    row_min = mean_matrix.min(axis=1, keepdims=True)
    normalized_matrix = (mean_matrix - row_min) / (row_max - row_min + 1e-6)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(normalized_matrix, aspect='auto', cmap='viridis')

    ax.set_xticks(range(n_stages))
    ax.set_xticklabels([f'State{s + 1}' for s in unique_stages])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=9)

    plt.colorbar(im, ax=ax)
    plt.title(f"Тепловая карта признаков по состояниям - {seismogram_idx}", fontsize=14)
    plt.tight_layout()
    return fig


def check_stage_match_expert(st_edges, step_duration, expert_sample, fs=FS, tolerance_sec=1.0):
    if np.isnan(expert_sample):
        return None

    expert_time = expert_sample / fs
    stage_boundary_times = st_edges * step_duration

    for b_time in stage_boundary_times:
        if abs(b_time - expert_time) <= tolerance_sec:
            return True

    return False


def process_single_seismogram(waveform, metadata, seismogram_idx, output_subdir):
    print(f"Анализ сейсмограммы {metadata.get('trace_name', seismogram_idx)}")
    extractor = FeatureExtractor(FS, EPOCH_DURATION, OVERLAP)
    features, feature_names = extractor.extract(waveform)

    if np.isnan(features).any():
        features = np.nan_to_num(features)

    best_sol, feats_pca, results_df = run_sda_alg(features)
    st_edges = np.array(best_sol['St_edges'])
    n_stages = len(st_edges) - 1
    labels = np.zeros(features.shape[0], dtype=int)

    for i in range(n_stages):
        labels[st_edges[i]:st_edges[i + 1]] = i

    print(f"   - Найдено стадий: {n_stages} ")
    durations = [(st_edges[i + 1] - st_edges[i]) * STEP_DURATION for i in range(n_stages)]
    print(f"   - Длительности: {[f'{d:.1f}' for d in durations]} сек ")

    clust_metrics = calculate_metrics_for_states(feats_pca, st_edges)
    print(f"   - Silhouette: {clust_metrics['avg_silhouette']:.4f} ")
    print(f"   - CH Index: {clust_metrics['avg_calinski_harabasz']:.2f} ")

    df_IV, iv_res = analyze_IV(features, labels, feature_names)
    print(f"   - Mean IV: {iv_res['mean_IV']:.3f} ")
    print(f"   - Признаков с IV≥0.4: {iv_res['pct_features_IV_ge_0.4']:.1f}% ")

    ratio = np.nan
    features_surr = feats_pca.copy()
    np.random.shuffle(features_surr)

    best_surr, _, _ = run_sda_alg(features_surr)

    if best_surr:
        st_edges_surr = np.array(best_surr['St_edges'])
        surr_metrics = calculate_metrics_for_states(features_surr, st_edges_surr)

        labels_surr = np.zeros(len(features), dtype=int)
        for i in range(len(st_edges_surr) - 1):
            labels_surr[st_edges_surr[i]:st_edges_surr[i + 1]] = i

        df_IV_surr, _ = analyze_IV(features, labels_surr, feature_names)

        ratio = clust_metrics['avg_silhouette'] / (surr_metrics['avg_silhouette'] + 1e-6)
        print(f"   - Surrogate Ratio: {ratio:.2f}x ")
    else:
        surr_metrics = {}
        df_IV_surr = None

    p_match = check_stage_match_expert(st_edges, STEP_DURATION,
                                       metadata.get('trace_P_arrival_sample', np.nan))
    s_match = check_stage_match_expert(st_edges, STEP_DURATION,
                                       metadata.get('trace_S_arrival_sample', np.nan))

    print(f"   - Совпадение с P-вступлением: {p_match if p_match is not None else 'нет метки'} ")
    print(f"   - Совпадение с S-вступлением: {s_match if s_match is not None else 'нет метки'} ")

    fig1 = plot_seismogram_with_stages(waveform, metadata, st_edges, seismogram_idx, STEP_DURATION)
    fig1.savefig(os.path.join(output_subdir, f'seismogram_{seismogram_idx}_stages.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig1)

    fig2 = plot_physical_features_analysis(features, feature_names, labels, seismogram_idx)
    fig2.savefig(os.path.join(output_subdir, f'seismogram_{seismogram_idx}_features.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)

    fig3 = plot_clustering_metrics_comparison(clust_metrics, surr_metrics, seismogram_idx)
    fig3.savefig(os.path.join(output_subdir, f'seismogram_{seismogram_idx}_metrics.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig3)

    fig4 = plot_IV_analysis(df_IV, df_IV_surr, seismogram_idx)
    fig4.savefig(os.path.join(output_subdir, f'seismogram_{seismogram_idx}_IV.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig4)

    fig5 = plot_feature_importance_heatmap(features, feature_names, labels, seismogram_idx)
    fig5.savefig(os.path.join(output_subdir, f'seismogram_{seismogram_idx}_heatmap.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig5)

    features_df = pd.DataFrame(features, columns=feature_names)
    features_df['stage'] = labels

    return {
        'trace_name': metadata.get('trace_name', ''),
        'magnitude': metadata.get('source_magnitude', 0),
        'distance_km': metadata.get('source_distance_km', 0),
        'n_stages': n_stages,
        'silhouette': clust_metrics['avg_silhouette'],
        'ch_index': clust_metrics['avg_calinski_harabasz'],
        'mean_iv': iv_res['mean_IV'],
        'stage_durations': durations,
        'p_detected': p_match,
        's_detected': s_match,
        'surrogate_ratio': ratio,
        'features': features,
        'st_edges': st_edges,
        'feature_names': feature_names
    }


def calculate_IV_binary(features, mask, n_bins=10):
    n_epochs, n_features = features.shape
    iv_values = np.zeros(n_features)

    for f_idx in range(n_features):
        f_val = features[:, f_idx]
        bins = np.percentile(f_val, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)

        if len(bins) < 2:
            continue

        digitized = np.digitize(f_val, bins) - 1
        p_good_global = np.mean(mask)
        p_bad_global = 1 - p_good_global
        iv_cls = 0.0

        for b_val in np.unique(digitized):
            bin_mask = digitized == b_val

            if np.sum(bin_mask) == 0:
                continue

            p_good = np.sum(mask[bin_mask]) / np.sum(bin_mask)
            p_bad = 1 - p_good

            if p_good > 0 and p_bad > 0 and p_good_global > 0:
                woe = np.log((p_good / p_bad) / (p_good_global / p_bad_global))
                iv_cls += (p_good - p_bad) * woe

        iv_values[f_idx] = iv_cls

    return iv_values


def main():
    loader = Loader(DATA_PATH, EVENTS_METADATA_PATH, SAMPLE_SIZE, random_seed=RANDOM_SEED)
    waveforms, metadata_df = loader.load_data()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_OUTPUT_DIR, f"{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    all_results = []

    for i, waveform in enumerate(waveforms):
        metadata = metadata_df.iloc[i]

        trace_name_clean = str(metadata['trace_name']).replace('/', '_').replace('\\', '_')
        event_dir = os.path.join(run_dir, f"event_{i}_{trace_name_clean}")
        os.makedirs(event_dir, exist_ok=True)

        result = process_single_seismogram(waveform, metadata, i, event_dir)
        all_results.append(result)

    df_res = pd.DataFrame(all_results)

    summary_path = os.path.join(run_dir, 'summary_results.csv')
    df_res.to_csv(summary_path, index=False)

    print("\nИТОГОВАЯ СТАТИСТИКА:")
    print(f"   - Среднее количество стадий: {df_res['n_stages'].mean():.1f} ± {df_res['n_stages'].std():.1f} ")
    print(f"   - Средний Silhouette: {df_res['silhouette'].mean():.3f} ± {df_res['silhouette'].std():.3f} ")
    print(f"   - Средний CH Index: {df_res['ch_index'].mean():.2f} ± {df_res['ch_index'].std():.2f} ")
    print(f"   - Средний IV: {df_res['mean_iv'].mean():.3f} ± {df_res['mean_iv'].std():.3f} ")

    all_durations = []
    for durations in df_res['stage_durations']:
        all_durations.extend(durations)
    print(f"   - Средняя длительность стадии: {np.mean(all_durations):.2f} ± {np.std(all_durations):.2f} сек ")

    avg_ratio = df_res['surrogate_ratio'].dropna().mean()
    n_valid_ratio = df_res['surrogate_ratio'].notna().sum()
    print(
        f"   - Средний прирост относительно суррогата (Real/Surrogate Ratio): {avg_ratio:.2f}x (на {n_valid_ratio} событиях) ")

    p_valid = df_res['p_detected'].notna()
    s_valid = df_res['s_detected'].notna()

    p_accuracy = df_res.loc[p_valid, 'p_detected'].mean() if p_valid.any() else np.nan
    s_accuracy = df_res.loc[s_valid, 's_detected'].mean() if s_valid.any() else np.nan

    print(f"   - Точность детектирования P-волны: {p_accuracy:.2%} (на {p_valid.sum()} событиях) ")
    print(f"   - Точность детектирования S-волны: {s_accuracy:.2%} (на {s_valid.sum()} событиях) ")

    print(f"Результаты сохранены в: {run_dir} ")

    return df_res

if __name__ == "__main__":
    main()
