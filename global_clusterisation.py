import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.feature_selection import f_classif
import datetime
import shutil

from state_detection import (
    Loader,
    calculate_IV_binary,
    process_single_seismogram
)

SAMPLE_SIZE = 30
FS = 100.0
EPOCH_DURATION = 0.8
OVERLAP = 0.5
STEP_DURATION = EPOCH_DURATION * (1 - OVERLAP)

BASE_PATH = "/Users/artyom/Desktop/Кактус/проекты/4 курс/Instance_sample_dataset_v3"
DATA_PATH = os.path.join(BASE_PATH, "data", "Instance_noise_1k.hdf5")
EVENTS_METADATA_PATH = os.path.join(BASE_PATH, "metadata", "metadata_Instance_noise_1k.csv")

BASE_OUTPUT_DIR = "./runs"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
RANDOM_SEED = 42


def plot_letter_feature_boxplots(features_z, feature_names, stage_labels, n_stage_clusters, run_dir):
    os.makedirs(os.path.join(run_dir, 'letter_analysis'), exist_ok=True)
    letters = [chr(ord('A') + k) for k in range(n_stage_clusters)]
    groups = {
        'Polarization': ['rectilinearity', 'planarity', 'log_energy_polar'],
        'Spectral': [n for n in feature_names if
                     'energy_' in n or 'rel_power_' in n or 'centroid' in n or 'kurtosis' in n],
        'STA_LTA': [n for n in feature_names if 'sta_lta' in n],
        'S_P': [n for n in feature_names if 'S_over_P' in n],
        'RMS': [n for n in feature_names if 'rms' in n],
        'Log_Energy': [n for n in feature_names if 'log_energy' in n and 'polar' not in n]
    }
    for group_name, group_feats in groups.items():
        indices = [i for i, n in enumerate(feature_names) if n in group_feats]
        if len(indices) == 0:
            continue
        n_cols = min(4, len(indices))
        n_rows = int(np.ceil(len(indices) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for i, idx in enumerate(indices):
            ax = axes[i]
            data = [features_z[stage_labels == k, idx] for k in range(n_stage_clusters)]
            ax.boxplot(data, tick_labels=letters)
            ax.set_title(feature_names[idx], fontsize=8)
            ax.grid(True, alpha=0.3)
        for j in range(len(indices), len(axes)):
            axes[j].axis('off')
        fig.suptitle(f'Распределение признаков по буквам: {group_name}', fontsize=12)
        plt.tight_layout()
        safe_group = group_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        fig.savefig(os.path.join(run_dir, 'letter_analysis', f'boxplot_{safe_group}.png'), dpi=150)
        plt.close(fig)


def plot_letter_feature_heatmap(features_z, feature_names, stage_labels, n_stage_clusters, run_dir):
    variances = np.var(features_z, axis=0)
    top_indices = np.argsort(variances)[-20:][::-1]
    top_feature_names = [feature_names[i] for i in top_indices]

    letters = [chr(ord('A') + k) for k in range(n_stage_clusters)]
    mean_matrix = np.zeros((len(top_feature_names), n_stage_clusters))

    for i, feat_idx in enumerate(top_indices):
        for j in range(n_stage_clusters):
            mean_matrix[i, j] = np.mean(features_z[stage_labels == j, feat_idx])

    row_min = mean_matrix.min(axis=1, keepdims=True)
    row_max = mean_matrix.max(axis=1, keepdims=True)
    norm_mat = (mean_matrix - row_min) / (row_max - row_min + 1e-6)

    fig, ax = plt.subplots(figsize=(max(8, n_stage_clusters * 1.5), len(top_feature_names) * 0.4))
    im = ax.imshow(norm_mat, aspect='auto', cmap='viridis')
    ax.set_xticks(range(n_stage_clusters))
    ax.set_xticklabels(letters)
    ax.set_yticks(range(len(top_feature_names)))
    ax.set_yticklabels(top_feature_names, fontsize=9)
    plt.colorbar(im, ax=ax)
    plt.title('ТОП-20 признаков глобальной кластеризации', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(run_dir, 'letter_analysis', 'heatmap_features.png'), dpi=150)
    return fig


def plot_letter_IV_analysis(features_z, stage_labels, feature_names, n_stage_clusters, run_dir):
    letters = [chr(ord('A') + k) for k in range(n_stage_clusters)]
    iv_dict = {}
    for k in range(n_stage_clusters):
        mask = (stage_labels == k)
        iv_vec = calculate_IV_binary(features_z, mask)
        iv_clipped = np.clip(iv_vec, 0, 1)
        iv_dict[k] = iv_clipped
    iv_table = pd.DataFrame({letters[k]: iv_dict[k] for k in range(n_stage_clusters)}, index=feature_names)
    iv_table.to_csv(os.path.join(run_dir, 'letter_analysis', 'IV_by_letter.csv'))
    fig, axes = plt.subplots(1, n_stage_clusters, figsize=(4 * n_stage_clusters, 8))
    if n_stage_clusters == 1:
        axes = [axes]
    for k, ax in enumerate(axes):
        iv_series = pd.Series(iv_dict[k], index=feature_names).sort_values(ascending=False)
        top_iv = iv_series.head(10)
        ax.barh(top_iv.index[::-1], top_iv.values[::-1], color='steelblue')
        ax.set_title(f'Cluster {letters[k]}', fontsize=12)
        ax.set_xlabel('IV')
        ax.grid(True, alpha=0.3)
    fig.suptitle('Топ-10 признаков по информационной ценности (IV) для каждой буквы', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(run_dir, 'letter_analysis', 'IV_top_features.png'), dpi=150)
    return fig


def cluster_stages(all_stage_vectors, n_clusters_range=(3, 5)):
    X = np.array(all_stage_vectors)
    scaler = RobustScaler(quantile_range=(25, 75))
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(15, X.shape[0] - 1, X.shape[1]), whiten=True)
    X_pca = pca.fit_transform(X_scaled)

    results = {}

    for n_cl in range(n_clusters_range[0], n_clusters_range[1] + 1):
        km = KMeans(n_clusters=n_cl, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_pca)

        if len(np.unique(labels)) < 2:
            continue

        sil = silhouette_score(X_pca, labels)
        ch = calinski_harabasz_score(X_pca, labels)
        db = davies_bouldin_score(X_pca, labels)

        results[n_cl] = {
            'model': km,
            'labels': labels,
            'silhouette': sil,
            'calinski_harabasz': ch,
            'davies_bouldin': db
        }

    sorted_by_sil = sorted(results.keys(), key=lambda k: results[k]['silhouette'], reverse=True)
    sorted_by_ch = sorted(results.keys(), key=lambda k: results[k]['calinski_harabasz'], reverse=True)
    sorted_by_db = sorted(results.keys(), key=lambda k: results[k]['davies_bouldin'])

    rank_sil = {n: i + 1 for i, n in enumerate(sorted_by_sil)}
    rank_ch = {n: i + 1 for i, n in enumerate(sorted_by_ch)}
    rank_db = {n: i + 1 for i, n in enumerate(sorted_by_db)}

    total_ranks = {n: rank_sil[n] + rank_ch[n] + rank_db[n] for n in results}
    best_n_rank = min(total_ranks, key=total_ranks.get)

    n_list = sorted(results.keys())
    ch_values = [results[n]['calinski_harabasz'] for n in n_list]
    local_peak_n = None

    for i in range(1, len(n_list) - 1):
        if ch_values[i] > ch_values[i - 1] and ch_values[i] > ch_values[i + 1]:
            local_peak_n = n_list[i]
            break

    if local_peak_n is None:
        mid = (n_clusters_range[0] + n_clusters_range[1]) // 2
        local_peak_n = min(results.keys(), key=lambda x: abs(x - mid))

    best_n = best_n_rank
    if best_n_rank == n_clusters_range[0] or best_n_rank == n_clusters_range[1]:
        if local_peak_n != best_n_rank and local_peak_n in results:
            max_ch = max(ch_values)
            ch_peak = results[local_peak_n]['calinski_harabasz']
            if ch_peak >= 0.9 * max_ch:
                best_n = local_peak_n

    best_labels = results[best_n]['labels']
    best_model = results[best_n]['model']
    best_sil = results[best_n]['silhouette']

    print(f"Оптимальное число кластеров: {best_n}, "
          f"силуэт: {best_sil:.3f}, "
          f"CH: {results[best_n]['calinski_harabasz']:.2f}, "
          f"DB: {results[best_n]['davies_bouldin']:.3f}")
    return best_model, best_labels, best_n, X_pca, pca


def make_event_strings(event_stage_vectors, model, scaler, pca):
    event_strings = []

    for stage_vecs in event_stage_vectors:
        if not stage_vecs:
            event_strings.append("")
            continue

        stacked = np.array(stage_vecs)
        scaled = scaler.transform(stacked)
        pca_rep = pca.transform(scaled)
        labels = model.predict(pca_rep)
        letters = [chr(ord('A') + lbl) for lbl in labels]
        merged = []

        for ch in letters:
            if not merged or ch != merged[-1]:
                merged.append(ch)

        event_strings.append("".join(merged))

    return event_strings


def main():
    loader = Loader(DATA_PATH, EVENTS_METADATA_PATH, SAMPLE_SIZE, random_seed=RANDOM_SEED)
    waveforms, metadata_df = loader.load_data()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_OUTPUT_DIR, f"{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    all_results_raw = []

    for i, waveform in enumerate(waveforms):
        metadata = metadata_df.iloc[i]
        trace_name_clean = str(metadata['trace_name']).replace('/', '_').replace('\\', '_')
        event_dir = os.path.join(run_dir, f"event_{i}_{trace_name_clean}")
        os.makedirs(event_dir, exist_ok=True)
        res = process_single_seismogram(waveform, metadata, i, event_dir)
        if res:
            all_results_raw.append(res)

    all_epochs = np.vstack([res['features'] for res in all_results_raw])
    global_mean = np.mean(all_epochs, axis=0)
    global_std = np.std(all_epochs, axis=0)
    global_std[global_std == 0] = 1.0
    feature_names = all_results_raw[0]['feature_names']

    global_combined_vectors = []
    event_combined_vectors = []
    final_results = []

    for res in all_results_raw:
        features_z = (res['features'] - global_mean) / global_std
        combined_list = []

        for s in range(len(res['st_edges']) - 1):
            start, end = res['st_edges'][s], res['st_edges'][s + 1]
            mask = np.zeros(features_z.shape[0], dtype=bool)
            mask[start:end] = True
            median_z = np.median(features_z[mask], axis=0)
            combined_list.append(median_z)

        event_combined_vectors.append(combined_list)
        global_combined_vectors.extend(combined_list)

        final_results.append({
            'trace_name': res['trace_name'],
            'magnitude': res['magnitude'],
            'distance_km': res['distance_km'],
            'n_stages': res['n_stages'],
            'silhouette': res['silhouette'],
            'ch_index': res['ch_index'],
            'mean_iv': res['mean_iv'],
            'stage_durations': res['stage_durations'],
            'surrogate_ratio': res['surrogate_ratio'],
            'combined_stage_vectors_list': combined_list
        })

    df_res = pd.DataFrame(final_results)

    print("\nИТОГОВАЯ СТАТИСТИКА:")
    combined_feature_names = [f'median_{n}' for n in feature_names]
    X_stages = np.array(global_combined_vectors)

    stage_model, stage_labels, n_stage_clusters, stage_pca, stage_pca_obj = cluster_stages(
        global_combined_vectors)

    F_scores, p_values = f_classif(X_stages, stage_labels)

    importance_df = pd.DataFrame({
        'feature': combined_feature_names,
        'F_score': F_scores,
        'p_value': p_values
    }).sort_values('F_score', ascending=False)

    print("\nТОП-15 признаков:")
    print(importance_df.head(15).to_string(index=False))

    all_features_z = np.vstack([(res['features'] - global_mean) / global_std for res in all_results_raw])
    epoch_letter_labels = np.zeros(all_features_z.shape[0], dtype=int)
    offset = 0

    for idx, res in enumerate(all_results_raw):
        n_epochs = res['features'].shape[0]
        stage_vecs = event_combined_vectors[idx]
        stacked = np.array(stage_vecs)
        scaler = RobustScaler(quantile_range=(25, 75)).fit(np.array(global_combined_vectors))
        scaled = scaler.transform(stacked)
        pca_rep = stage_pca_obj.transform(scaled)
        letter_preds = stage_model.predict(pca_rep)

        for s, (start, end) in enumerate(zip(res['st_edges'][:-1], res['st_edges'][1:])):
            epoch_letter_labels[offset + start: offset + end] = letter_preds[s]

        offset += n_epochs

    plot_letter_feature_boxplots(all_features_z, feature_names, epoch_letter_labels, n_stage_clusters, run_dir)
    plot_letter_feature_heatmap(all_features_z, feature_names, epoch_letter_labels, n_stage_clusters, run_dir)
    plot_letter_IV_analysis(all_features_z, epoch_letter_labels, feature_names, n_stage_clusters, run_dir)

    scaler = RobustScaler(quantile_range=(25, 75)).fit(np.array(global_combined_vectors))
    event_strings = make_event_strings(event_combined_vectors, stage_model, scaler, stage_pca_obj)
    df_res['stage_string'] = event_strings

    stage_seq_dir = os.path.join(run_dir, "stage_sequences")
    os.makedirs(stage_seq_dir, exist_ok=True)

    for idx, row in df_res.iterrows():
        seq = row['stage_string']
        safe_seq = seq.replace('/', '_').replace('\\', '_').replace(':', '_')
        seq_folder = os.path.join(stage_seq_dir, safe_seq)
        os.makedirs(seq_folder, exist_ok=True)
        src_file = os.path.join(run_dir, f"event_{idx}_{row['trace_name'].replace('/', '_')}",
                                f"seismogram_{idx}_stages.png")
        if os.path.exists(src_file):
            shutil.copy(src_file, os.path.join(seq_folder, f"{row['trace_name'].replace('/', '_')}_stages.png"))

    print(f"Результаты сохранены в: {run_dir}")
    return df_res


if __name__ == "__main__":
    main()
