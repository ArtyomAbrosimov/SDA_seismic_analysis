import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import ruptures as rpt
from hmmlearn import hmm

from state_detection import (
    Loader,
    FeatureExtractor,
    run_sda_alg,
    check_stage_match_expert,
    calculate_metrics_for_states,
    analyze_IV,
    FS, EPOCH_DURATION, OVERLAP, STEP_DURATION
)

SAMPLE_SIZE = 30
BASE_PATH = "/Users/artyom/Desktop/Кактус/проекты/4 курс/Instance_sample_dataset_v3"
DATA_PATH = os.path.join(BASE_PATH, "data", "Instance_events_gm_10k.hdf5")
METADATA_PATH = os.path.join(BASE_PATH, "metadata", "metadata_Instance_events_10k.csv")

BASE_OUTPUT_DIR = "./comparison_results"
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"{RUN_TIMESTAMP}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STAGE_COLORS = plt.cm.Set3(np.linspace(0, 1, 12))
RANDOM_SEED = 42

PLOT_METHODS = ['SDA', 'STA/LTA', 'PELT', 'HMM']


def consolidate_stages(boundaries, n_epochs, min_epochs=3, max_stages=5):
    if len(boundaries) == 0:
        return []

    edges = sorted([e for e in set(boundaries) if 0 < e < n_epochs])
    clean = []
    last_pos = 0

    for e in edges:
        if e - last_pos >= min_epochs:
            clean.append(e)
            last_pos = e

    full_edges = [0] + clean + [n_epochs]

    while len(full_edges) - 1 > max_stages and len(full_edges) > 2:
        lengths = np.diff(full_edges)
        min_idx = np.argmin(lengths)

        if min_idx == 0:
            del_idx = 1
        elif min_idx == len(full_edges) - 2:
            del_idx = min_idx
        else:
            del_idx = min_idx if lengths[min_idx - 1] > lengths[min_idx + 1] else min_idx + 1

        del full_edges[del_idx]

    return full_edges[1:-1]


def run_sta_lta(waveform, fs, n_epochs):
    step_samples = int(EPOCH_DURATION * (1 - OVERLAP) * fs)
    env = np.sqrt(np.sum(waveform ** 2, axis=0))
    sta_len, lta_len = int(1.0 * fs), int(10.0 * fs)
    sta = np.convolve(env, np.ones(sta_len) / sta_len, mode='same')
    lta = np.convolve(env, np.ones(lta_len) / lta_len, mode='same')
    lta[lta == 0] = 1e-12
    cft = sta / lta
    thr = np.percentile(cft, 85) * 1.3
    above = np.where(cft > thr)[0]

    if len(above) < int(2 * fs):
        return consolidate_stages([n_epochs // 3, 2 * n_epochs // 3], n_epochs)

    start, end = above[0], above[-1]
    segment_energy = np.array(
        [np.mean(env[i:i + max(int(0.2 * fs), 5)]) for i in range(start, end, max(int(0.2 * fs), 5))])
    cum_e = np.cumsum(segment_energy)
    total_e = cum_e[-1]

    if total_e <= 0:
        return [start // step_samples, end // step_samples]

    t30 = start + int(np.searchsorted(cum_e, 0.3 * total_e) * max(int(0.2 * fs), 5))
    t70 = start + int(np.searchsorted(cum_e, 0.7 * total_e) * max(int(0.2 * fs), 5))
    t90 = start + int(np.searchsorted(cum_e, 0.9 * total_e) * max(int(0.2 * fs), 5))
    epoch_bounds = [b // step_samples for b in [t30, t70, t90]]

    return consolidate_stages(epoch_bounds, n_epochs)


def run_pelt(features_pca, penalty=15):
    n_epochs = features_pca.shape[0]
    algo = rpt.Pelt(model="l2", min_size=3).fit(features_pca)
    bkps = algo.predict(pen=penalty)
    return consolidate_stages(bkps[:-1], n_epochs)


def run_hmm(features_pca, n_states=4):
    n_epochs = features_pca.shape[0]
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, random_state=RANDOM_SEED,
                            tol=1e-3)
    model.fit(features_pca)
    raw_labels = model.predict(features_pca)
    smooth_labels = median_filter(raw_labels, size=5)
    boundaries = np.where(np.diff(smooth_labels))[0] + 1
    return consolidate_stages(boundaries, n_epochs)


def run_sda_wrapper(raw_features):
    best, _, _ = run_sda_alg(raw_features)
    return consolidate_stages(best.get('St_edges', []), raw_features.shape[0])


METHODS = {
    'SDA': lambda w, f, m, r: run_sda_wrapper(r),
    'STA/LTA': lambda w, f, m, r: run_sta_lta(w, FS, f.shape[0]),
    'PELT': lambda w, f, m, r: run_pelt(f, penalty=15),
    'HMM': lambda w, f, m, r: run_hmm(f, n_states=4),
}


def plot_seismogram_stages(waveform, metadata, st_edges, method_name, output_path):
    n_samples = waveform.shape[1]
    time_axis = np.arange(n_samples) / FS
    n_stages = len(st_edges) - 1 if len(st_edges) > 1 else 1
    fig, axes = plt.subplots(4, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [3, 3, 3, 1]})
    channel_names = ['Z (Vertical)', 'N (North)', 'E (East)']
    colors = STAGE_COLORS[:max(n_stages, 1)]

    for ch_idx in range(3):
        ax = axes[ch_idx]
        ax.plot(time_axis, waveform[ch_idx], 'k-', linewidth=0.5, alpha=0.7)

        for i, (start, end) in enumerate(zip(st_edges[:-1], st_edges[1:])):
            t_start = start * STEP_DURATION
            t_end = end * STEP_DURATION
            ax.axvspan(t_start, t_end, alpha=0.2, color=colors[i % len(colors)])

        if 'trace_P_arrival_sample' in metadata and not np.isnan(metadata.get('trace_P_arrival_sample', np.nan)):
            ax.axvline(metadata['trace_P_arrival_sample'] / FS, color='r', ls='--', alpha=0.5,
                       label='P-wave' if ch_idx == 0 else '')
        if 'trace_S_arrival_sample' in metadata and not np.isnan(metadata.get('trace_S_arrival_sample', np.nan)):
            ax.axvline(metadata['trace_S_arrival_sample'] / FS, color='b', ls='--', alpha=0.5,
                       label='S-wave' if ch_idx == 0 else '')

        ax.set_ylabel(channel_names[ch_idx])
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, time_axis[-1])
        if ch_idx == 0:
            ax.legend(loc='upper right')

    ax_stage = axes[3]

    for i, (start, end) in enumerate(zip(st_edges[:-1], st_edges[1:])):
        t_start = start * STEP_DURATION
        t_end = end * STEP_DURATION
        width = t_end - t_start
        ax_stage.barh(0, width, left=t_start, height=0.5, color=colors[i % len(colors)], edgecolor='black', alpha=0.7)

        if width > 2.0:
            ax_stage.text((t_start + t_end) / 2, 0, f'S{i + 1}', ha='center', va='center', fontweight='bold')

    ax_stage.set_yticks([])
    ax_stage.set_xlabel('Время (с)', fontsize=11)
    ax_stage.set_xlim(0, time_axis[-1])

    trace_name = metadata.get('trace_name', 'unknown')
    plt.suptitle(f"{method_name} | {trace_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_comparison(waveform, metadata, boundaries_dict, output_path):
    fig = plt.figure(figsize=(36, 20))
    gs_big = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    for idx, m_name in enumerate(PLOT_METHODS):
        i, j = divmod(idx, 2)
        gs_inner = gs_big[i, j].subgridspec(4, 1, hspace=0.05, height_ratios=[3, 3, 3, 1])
        axes = [fig.add_subplot(gs_inner[k]) for k in range(4)]

        st_edges = boundaries_dict.get(m_name, [0, waveform.shape[1]])
        n_samples = waveform.shape[1]
        time_axis = np.arange(n_samples) / FS
        n_stages = len(st_edges) - 1 if len(st_edges) > 1 else 1
        channel_names = ['Z (Vertical)', 'N (North)', 'E (East)']
        colors = STAGE_COLORS[:max(n_stages, 1)]

        for ch_idx in range(3):
            ax = axes[ch_idx]
            ax.plot(time_axis, waveform[ch_idx], 'k-', linewidth=0.5, alpha=0.7)
            for k, (start, end) in enumerate(zip(st_edges[:-1], st_edges[1:])):
                t_start = start * STEP_DURATION
                t_end = end * STEP_DURATION
                ax.axvspan(t_start, t_end, alpha=0.2, color=colors[k % len(colors)])

            if 'trace_P_arrival_sample' in metadata and not np.isnan(metadata.get('trace_P_arrival_sample', np.nan)):
                ax.axvline(metadata['trace_P_arrival_sample'] / FS, color='r', ls='--', alpha=0.5,
                           label='P-wave' if ch_idx == 0 else '')
            if 'trace_S_arrival_sample' in metadata and not np.isnan(metadata.get('trace_S_arrival_sample', np.nan)):
                ax.axvline(metadata['trace_S_arrival_sample'] / FS, color='b', ls='--', alpha=0.5,
                           label='S-wave' if ch_idx == 0 else '')

            ax.set_ylabel(channel_names[ch_idx], fontsize=9)
            ax.grid(True, alpha=0.2)
            ax.set_xlim(0, time_axis[-1])
            ax.tick_params(labelbottom=False)
            if ch_idx == 0:
                ax.legend(loc='upper right')

        ax_stage = axes[3]
        for k, (start, end) in enumerate(zip(st_edges[:-1], st_edges[1:])):
            t_start = start * STEP_DURATION
            t_end = end * STEP_DURATION
            width = t_end - t_start
            ax_stage.barh(0, width, left=t_start, height=0.5, color=colors[k % len(colors)], edgecolor='black',
                          alpha=0.7)
            if width > 2.0:
                ax_stage.text((t_start + t_end) / 2, 0, f'S{k + 1}', ha='center', va='center', fontweight='bold')

        ax_stage.set_yticks([])
        ax_stage.set_xlabel('Время (с)', fontsize=10)
        ax_stage.set_xlim(0, time_axis[-1])
        axes[0].set_title(f"{m_name}", fontsize=24, pad=10)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def evaluate_method(features, features_pca, edges, labels, feature_names, metadata):
    n_stages = len(edges) - 1
    valid = n_stages >= 2
    res = {'n_stages': n_stages, 'valid': valid}

    if not valid:
        res.update({k: np.nan for k in ['silhouette', 'ch_index', 'ward_dist', 'mean_iv', 'p_match', 's_match']})
        return res
    clust = calculate_metrics_for_states(features_pca, edges)

    res.update({
        'silhouette': clust['avg_silhouette'],
        'ch_index': clust['avg_calinski_harabasz'],
        'ward_dist': clust['avg_ward_distance']
    })

    _, iv_res = analyze_IV(features, labels, feature_names)
    res['mean_iv'] = iv_res['mean_IV']

    p_exp = metadata.get('trace_P_arrival_sample', np.nan)
    s_exp = metadata.get('trace_S_arrival_sample', np.nan)

    p_match = check_stage_match_expert(edges, STEP_DURATION, p_exp)
    s_match = check_stage_match_expert(edges, STEP_DURATION, s_exp)

    res['p_match'] = int(p_match) if isinstance(p_match, bool) else np.nan
    res['s_match'] = int(s_match) if isinstance(s_match, bool) else np.nan

    return res


def main():
    loader = Loader(DATA_PATH, METADATA_PATH, SAMPLE_SIZE, random_seed=RANDOM_SEED)
    waveforms, metadata_df = loader.load_data()

    p_col = 'trace_P_arrival_sample'
    s_col = 'trace_S_arrival_sample'

    if p_col in metadata_df.columns and s_col in metadata_df.columns:
        valid_mask = (metadata_df[p_col].notna()) & (metadata_df[s_col].notna())
        valid_mask &= (metadata_df[p_col] > 0) & (metadata_df[s_col] > 0)
        metadata_df = metadata_df[valid_mask].reset_index(drop=True)
        waveforms = [w for w, is_valid in zip(waveforms, valid_mask) if is_valid]

    extractor = FeatureExtractor(FS, EPOCH_DURATION, OVERLAP)
    results = {m: [] for m in METHODS.keys()}

    for i in range(len(waveforms)):
        waveform = waveforms[i]
        meta = metadata_df.iloc[i]
        trace_name = str(meta.get('trace_name', f'trace_{i}')).replace('/', '_').replace('\\', '_')
        trace_dir = os.path.join(OUTPUT_DIR, trace_name)
        os.makedirs(trace_dir, exist_ok=True)
        features, feat_names = extractor.extract(waveform)

        if np.isnan(features).any():
            features = np.nan_to_num(features)

        features_scaled = RobustScaler().fit_transform(features)
        feats_pca = PCA(n_components=0.85, whiten=True).fit_transform(features_scaled)
        n_epochs = feats_pca.shape[0]
        boundaries_dict = {}

        for m_name in PLOT_METHODS:
            m_func = METHODS[m_name]
            boundaries = m_func(waveform, feats_pca, meta, features)
            boundaries = [0] + sorted(list(set(boundaries))) + [n_epochs]
            edges = np.unique(np.array(boundaries))
            labels = np.zeros(n_epochs, dtype=int)

            for j in range(len(edges) - 1):
                labels[edges[j]:edges[j + 1]] = j

            boundaries_dict[m_name] = edges

            res = evaluate_method(features, feats_pca, edges, labels, feat_names, meta)
            results[m_name].append(res)

            plot_path = os.path.join(trace_dir, f'stages_{m_name.replace("/", "_")}.png')
            plot_seismogram_stages(waveform, meta, edges, m_name, plot_path)

        plot_comparison(waveform, meta, boundaries_dict,
                        os.path.join(trace_dir, 'comparison.png'))

    print("\nИТОГОВАЯ СТАТИСТИКА ПО МЕТОДАМ")
    summary = []

    for m_name in METHODS.keys():
        df_m = pd.DataFrame(results[m_name])
        valid = df_m['valid'].sum()

        avg_stages = round(df_m.loc[df_m['valid'], 'n_stages'].mean(), 2) if valid > 0 else np.nan

        if valid > 0:
            valid_df = df_m[df_m['valid']]
            p_ok = (valid_df['p_match'] == 1)
            s_ok = (valid_df['s_match'] == 1)
            any_match = (p_ok | s_ok).mean()
            both_match = (p_ok & s_ok).mean()
        else:
            any_match = np.nan
            both_match = np.nan

        row = {
            'Метод': m_name,
            'Avg Stages': avg_stages,
            'Silhouette': round(df_m.loc[df_m['valid'], 'silhouette'].mean(), 3) if valid > 0 else np.nan,
            'CH Index': round(df_m.loc[df_m['valid'], 'ch_index'].mean(), 1) if valid > 0 else np.nan,
            'Ward Dist': round(df_m.loc[df_m['valid'], 'ward_dist'].mean(), 1) if valid > 0 else np.nan,
            'Mean IV': round(df_m.loc[df_m['valid'], 'mean_iv'].mean(), 3) if valid > 0 else np.nan,
            'P Accuracy': round(df_m['p_match'].dropna().mean(), 3) if df_m['p_match'].notna().any() else np.nan,
            'S Accuracy': round(df_m['s_match'].dropna().mean(), 3) if df_m['s_match'].notna().any() else np.nan,
            'Any Match': round(any_match, 3),
            'Both Match': round(both_match, 3)
        }
        summary.append(row)

        p_acc = f"{row['P Accuracy']:.1%}" if not np.isnan(row['P Accuracy']) else "N/A"
        s_acc = f"{row['S Accuracy']:.1%}" if not np.isnan(row['S Accuracy']) else "N/A"
        stages_str = f"{row['Avg Stages']:.1f}" if not np.isnan(row['Avg Stages']) else "N/A"
        any_str = f"{row['Any Match']:.1%}" if not np.isnan(row['Any Match']) else "N/A"
        both_str = f"{row['Both Match']:.1%}" if not np.isnan(row['Both Match']) else "N/A"

        print(f"{m_name:<8} | Stages:{stages_str:>6} | Sil:{row['Silhouette']:.3f} | CH:{row['CH Index']:>7.1f} | "
              f"Ward:{row['Ward Dist']:>8.1f} | IV:{row['Mean IV']:.3f} | "
              f"P:{p_acc:>6} | S:{s_acc:>6} | Any:{any_str:>6} | Both:{both_str:>6}")


if __name__ == "__main__":
    main()
