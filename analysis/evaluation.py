# evaluations/metrics.py
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings("ignore")


# ------------------------
# Utility Functions
# ------------------------

def safe_discretize(data, n_bins=20):
    """Safely discretize continuous data"""
    if len(np.unique(data)) <= n_bins:
        return data.astype(int)
    else:
        try:
            return np.digitize(data, np.histogram(data, bins=n_bins)[1][:-1])
        except:
            # Fallback to simple binning
            return ((data - data.min()) / (data.max() - data.min() + 1e-8) * (n_bins - 1)).astype(int)

def handle_labels_input(labels):
    """Handle different label formats (dict with 'values'/'classes' or direct tensor)"""
    if isinstance(labels, dict):
        # Use discrete classes for most metrics, continuous values for correlation-based ones
        if "classes" in labels:
            discrete_labels = labels["classes"]
            continuous_labels = labels.get("values", labels["classes"])
        else:
            discrete_labels = labels["values"]
            continuous_labels = labels["values"]
    else:
        # Assume it's a tensor/array
        discrete_labels = labels
        continuous_labels = labels
    
    return discrete_labels, continuous_labels


# ------------------------
# Core Metrics (CORRECTED)
# ------------------------

def modularity_index(representations, labels):
    """
    Modularity index: For each latent dim, compute MI with each factor.
    A neuron is modular if its MI is concentrated on one factor.
    """
    try:
        reps = representations.detach().cpu().numpy() if torch.is_tensor(representations) else representations
        discrete_labels, _ = handle_labels_input(labels)
        labs = discrete_labels.detach().cpu().numpy() if torch.is_tensor(discrete_labels) else discrete_labels
        
        # Skip color factor (index 0) as it's constant in DSprites
        factor_start_idx = 1 if labs.shape[1] > 5 else 0
        n_factors = labs.shape[1] - factor_start_idx

        scores = []
        for i in range(reps.shape[1]):  # for each latent dimension
            mi_per_factor = []
            for f in range(factor_start_idx, labs.shape[1]):  # for each factor
                # Discretize latent dimension and factor
                dim_disc = safe_discretize(reps[:, i])
                lab_disc = labs[:, f].astype(int) if labs[:, f].dtype != int else labs[:, f]
                
                # Compute mutual information
                mi = normalized_mutual_info_score(lab_disc, dim_disc)
                mi_per_factor.append(mi)
            
            # Modularity: max MI / sum of all MIs (higher = more modular)
            total_mi = sum(mi_per_factor) + 1e-8
            if total_mi > 1e-6:
                modularity = max(mi_per_factor) / total_mi
                scores.append(modularity)
        
        return float(np.mean(scores)) if scores else 0.0
    except Exception as e:
        print(f"Error in modularity_index: {e}")
        return 0.0


def mutual_information(representations, labels):
    """
    Average MI between representation clusters and ground truth factors.
    """
    try:
        reps = representations.detach().cpu().numpy() if torch.is_tensor(representations) else representations
        discrete_labels, _ = handle_labels_input(labels)
        labs = discrete_labels.detach().cpu().numpy() if torch.is_tensor(discrete_labels) else discrete_labels
        
        # Use appropriate number of clusters
        n_samples = reps.shape[0]
        n_clusters = min(10, max(3, n_samples // 50))
        
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(reps)
        cluster_labels = km.labels_

        scores = []
        factor_start_idx = 1 if labs.shape[1] > 5 else 0
        
        for f in range(factor_start_idx, labs.shape[1]):
            factor_labels = labs[:, f].astype(int)
            mi = normalized_mutual_info_score(factor_labels, cluster_labels)
            scores.append(mi)
        
        return float(np.mean(scores))
    except Exception as e:
        print(f"Error in mutual_information: {e}")
        return 0.0


def rsa(representations, labels):
    """
    Representational Similarity Analysis: correlation between representation 
    distances and factor distances.
    """
    try:
        reps = representations.detach().cpu().numpy() if torch.is_tensor(representations) else representations
        _, continuous_labels = handle_labels_input(labels)
        labs = continuous_labels.detach().cpu().numpy() if torch.is_tensor(continuous_labels) else continuous_labels
        
        # Subsample for computational efficiency
        if reps.shape[0] > 1000:
            indices = np.random.choice(reps.shape[0], 1000, replace=False)
            reps = reps[indices]
            labs = labs[indices]
        
        # Compute representation distances
        rep_distances = pdist(reps, metric='cosine')
        
        scores = []
        factor_start_idx = 1 if labs.shape[1] > 5 else 0
        
        for f in range(factor_start_idx, labs.shape[1]):
            # Compute factor distances
            factor_distances = pdist(labs[:, f].reshape(-1, 1), metric='euclidean')
            
            # Compute correlation between distance matrices
            if len(np.unique(factor_distances)) > 1 and len(np.unique(rep_distances)) > 1:
                corr, _ = spearmanr(rep_distances, factor_distances)
                if not np.isnan(corr):
                    scores.append(abs(corr))  # Take absolute value
        
        return float(np.mean(scores)) if scores else 0.0
    except Exception as e:
        print(f"Error in rsa: {e}")
        return 0.0


def disentanglement_score(representations, labels):
    """
    FactorVAE-style disentanglement: each dimension should have low variance 
    within factor values (be informative about one factor).
    """
    try:
        reps = representations.detach().cpu().numpy() if torch.is_tensor(representations) else representations
        discrete_labels, continuous_labels = handle_labels_input(labels)
        
        # Use continuous labels for variance calculation
        labs = continuous_labels.detach().cpu().numpy() if torch.is_tensor(continuous_labels) else continuous_labels
        discrete_labs = discrete_labels.detach().cpu().numpy() if torch.is_tensor(discrete_labels) else discrete_labels
        
        factor_start_idx = 1 if labs.shape[1] > 5 else 0
        scores = []
        
        for i in range(reps.shape[1]):  # for each latent dimension
            factor_scores = []
            
            for f in range(factor_start_idx, labs.shape[1]):  # for each factor
                # Group by discrete factor values and compute within-group variance
                unique_vals = np.unique(discrete_labs[:, f])
                variances = []
                
                for val in unique_vals:
                    mask = discrete_labs[:, f] == val
                    if np.sum(mask) > 1:  # Need at least 2 samples
                        var = np.var(reps[mask, i])
                        variances.append(var)
                
                if variances:
                    avg_within_var = np.mean(variances)
                    total_var = np.var(reps[:, i]) + 1e-8
                    # Lower within-group variance relative to total = better disentanglement
                    disentanglement = 1.0 - (avg_within_var / total_var)
                    factor_scores.append(max(0.0, disentanglement))
            
            if factor_scores:
                scores.append(max(factor_scores))  # Best factor for this dimension
        
        return float(np.mean(scores)) if scores else 0.0
    except Exception as e:
        print(f"Error in disentanglement_score: {e}")
        return 0.0


def activation_overlap(representations):
    """
    Cosine similarity between different latent dimensions (lower = more orthogonal).
    """
    try:
        reps = representations.detach() if torch.is_tensor(representations) else torch.tensor(representations)
        
        # Normalize representations
        reps_norm = F.normalize(reps, dim=0, eps=1e-8)
        
        # Compute cosine similarity matrix between dimensions
        similarity_matrix = torch.mm(reps_norm.t(), reps_norm)
        
        # Remove diagonal (self-similarities)
        mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool)
        off_diagonal = similarity_matrix[mask]
        
        return float(off_diagonal.abs().mean().item())
    except Exception as e:
        print(f"Error in activation_overlap: {e}")
        return 0.0


def specialization_index(representations, labels):
    """
    How specialized each neuron is to individual factors (correlation-based).
    """
    try:
        reps = representations.detach().cpu().numpy() if torch.is_tensor(representations) else representations
        _, continuous_labels = handle_labels_input(labels)
        labs = continuous_labels.detach().cpu().numpy() if torch.is_tensor(continuous_labels) else continuous_labels
        
        factor_start_idx = 1 if labs.shape[1] > 5 else 0
        scores = []
        
        for i in range(reps.shape[1]):  # for each latent dimension
            correlations = []
            
            for f in range(factor_start_idx, labs.shape[1]):  # for each factor
                # Compute correlation between latent dimension and factor
                corr, p_val = pearsonr(reps[:, i], labs[:, f])
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            
            if correlations:
                # Specialization = max correlation (best factor match)
                scores.append(max(correlations))
        
        return float(np.mean(scores)) if scores else 0.0
    except Exception as e:
        print(f"Error in specialization_index: {e}")
        return 0.0


def clustering_purity(representations, labels):
    """
    Purity of clusters formed by representations w.r.t. ground truth factors.
    """
    try:
        reps = representations.detach().cpu().numpy() if torch.is_tensor(representations) else representations
        discrete_labels, _ = handle_labels_input(labels)
        labs = discrete_labels.detach().cpu().numpy() if torch.is_tensor(discrete_labels) else discrete_labels
        
        # Use the most diverse factor (usually shape in DSprites)
        factor_idx = 1 if labs.shape[1] > 5 else 0
        true_labels = labs[:, factor_idx].astype(int)
        
        n_clusters = len(np.unique(true_labels))
        if n_clusters < 2:
            return 0.0
        
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(reps)
        cluster_labels = km.labels_
        
        # Compute purity
        purity = 0
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) > 0:
                # Find most common true label in this cluster
                cluster_true_labels = true_labels[cluster_mask]
                most_common = np.bincount(cluster_true_labels).argmax()
                purity += np.sum(cluster_true_labels == most_common)
        
        return float(purity / len(true_labels))
    except Exception as e:
        print(f"Error in clustering_purity: {e}")
        return 0.0


def robustness_score(model, representations, noise_level=0.1):
    """
    Robustness to noise in representations.
    """
    try:
        reps = representations.detach() if torch.is_tensor(representations) else torch.tensor(representations)
        
        # Add noise
        noise = torch.randn_like(reps) * noise_level
        perturbed_reps = reps + noise
        
        # Compute similarity between original and perturbed
        similarity = F.cosine_similarity(reps, perturbed_reps, dim=-1)
        return float(similarity.mean().item())
    except Exception as e:
        print(f"Error in robustness_score: {e}")
        return 0.0


def interpretability_score(representations, labels):
    """
    How interpretable the learned representations are (based on linear separability).
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        reps = representations.detach().cpu().numpy() if torch.is_tensor(representations) else representations
        discrete_labels, _ = handle_labels_input(labels)
        labs = discrete_labels.detach().cpu().numpy() if torch.is_tensor(discrete_labels) else discrete_labels
        
        scores = []
        factor_start_idx = 1 if labs.shape[1] > 5 else 0
        
        # Try to predict each factor from representations
        for f in range(factor_start_idx, labs.shape[1]):
            try:
                factor_labels = labs[:, f].astype(int)
                if len(np.unique(factor_labels)) > 1:  # Must have multiple classes
                    # Simple train/test split
                    n_train = int(0.8 * len(reps))
                    indices = np.random.permutation(len(reps))
                    train_idx, test_idx = indices[:n_train], indices[n_train:]
                    
                    # Train linear classifier
                    clf = LogisticRegression(max_iter=1000, random_state=42)
                    clf.fit(reps[train_idx], factor_labels[train_idx])
                    
                    # Test accuracy
                    pred = clf.predict(reps[test_idx])
                    acc = accuracy_score(factor_labels[test_idx], pred)
                    scores.append(acc)
            except:
                continue
        
        return float(np.mean(scores)) if scores else 0.0
    except Exception as e:
        print(f"Error in interpretability_score: {e}")
        return 0.0


def compactness_score(representations):
    """
    How compact the learned representations are (lower intra-cluster variance).
    """
    try:
        reps = representations.detach().cpu().numpy() if torch.is_tensor(representations) else representations
        
        # Compute mean distance from centroid
        centroid = np.mean(reps, axis=0)
        distances = np.linalg.norm(reps - centroid, axis=1)
        
        # Normalize by dimensionality
        compactness = 1.0 / (1.0 + np.mean(distances) / np.sqrt(reps.shape[1]))
        return float(compactness)
    except Exception as e:
        print(f"Error in compactness_score: {e}")
        return 0.0


# ------------------------
# Master Evaluation (CORRECTED)
# ------------------------

def evaluate_all_metrics(model, representations, labels, device="cpu"):
    """
    Evaluate all modularity and disentanglement metrics.
    
    Args:
        model: The neural network model
        representations: Tensor of latent representations [N, latent_dim]
        labels: Dict with 'values' and 'classes' keys, or tensor [N, n_factors]
        device: Computing device
    
    Returns:
        Dictionary of all computed metrics
    """
    try:
        results = {}
        
        # Core modularity metrics
        results["modularity_index"] = modularity_index(representations, labels)
        results["mutual_information"] = mutual_information(representations, labels)
        results["disentanglement_score"] = disentanglement_score(representations, labels)
        results["specialization_index"] = specialization_index(representations, labels)
        
        # Representation quality metrics
        results["activation_overlap"] = activation_overlap(representations)
        results["robustness_score"] = robustness_score(model, representations)
        results["clustering_purity"] = clustering_purity(representations, labels)
        results["interpretability_score"] = interpretability_score(representations, labels)
        results["compactness_score"] = compactness_score(representations)
        
        # RSA (computationally expensive, so make it optional)
        try:
            results["rsa"] = rsa(representations, labels)
        except:
            results["rsa"] = 0.0
        
        # Compute composite scores
        modularity_metrics = ["modularity_index", "disentanglement_score", "specialization_index"]
        representation_metrics = ["mutual_information", "clustering_purity", "interpretability_score"]
        
        results["modularity_composite"] = np.mean([results[m] for m in modularity_metrics if m in results])
        results["representation_composite"] = np.mean([results[m] for m in representation_metrics if m in results])
        
        # Overall score (weighted combination)
        weights = {
            "modularity_composite": 0.4,
            "representation_composite": 0.3,
            "compactness_score": 0.2,
            "robustness_score": 0.1
        }
        
        results["total_score"] = sum(weights[k] * results.get(k, 0) for k in weights.keys())
        
        # Add metadata
        results["n_samples"] = representations.shape[0] if torch.is_tensor(representations) else len(representations)
        results["latent_dim"] = representations.shape[1] if torch.is_tensor(representations) else representations.shape[1]
        
        return results
        
    except Exception as e:
        print(f"Error in evaluate_all_metrics: {e}")
        return {"error": str(e), "total_score": 0.0}


# ------------------------
# Specialized Metrics for Different Experiments
# ------------------------

def evaluate_transfer_metrics(source_model, target_model, representations, labels):
    """Specific metrics for transfer learning experiments"""
    source_results = evaluate_all_metrics(source_model, representations, labels)
    target_results = evaluate_all_metrics(target_model, representations, labels)
    
    transfer_results = {
        "source_modularity": source_results.get("modularity_composite", 0),
        "target_modularity": target_results.get("modularity_composite", 0),
        "modularity_retention": target_results.get("modularity_composite", 0) / (source_results.get("modularity_composite", 1e-8) + 1e-8),
        "performance_gain": target_results.get("total_score", 0) - source_results.get("total_score", 0)
    }
    
    return {**target_results, **transfer_results}


def evaluate_multitask_metrics(representations, labels, task_outputs):
    """Specific metrics for multi-task learning experiments"""
    base_results = evaluate_all_metrics(None, representations, labels)
    
    # Add task-specific metrics if available
    if isinstance(task_outputs, dict):
        for task_name, outputs in task_outputs.items():
            if hasattr(outputs, 'shape') and len(outputs.shape) > 1:
                task_results = evaluate_all_metrics(None, outputs, labels)
                for key, val in task_results.items():
                    base_results[f"{task_name}_{key}"] = val
    
    return base_results

# ------------------------
# 🔹 Visual Diagnostics for Functional Modularity
# ------------------------
import os
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def visualize_functional_modularity(representations, labels, save_dir="results/visuals", prefix=""):
    """
    Generate visual diagnostics that help interpret functional modularity.
    Saves heatmaps, graph visualizations, and embedding projections.
    """
    os.makedirs(save_dir, exist_ok=True)

    reps = representations.detach().cpu().numpy() if torch.is_tensor(representations) else representations
    discrete_labels, _ = handle_labels_input(labels)
    labs = discrete_labels.detach().cpu().numpy() if torch.is_tensor(discrete_labels) else discrete_labels

    n_dims = reps.shape[1]
    prefix = prefix or "modularity"

    # ------------------------
    # 1️⃣ Neuron–Neuron Correlation Heatmap
    # ------------------------
    try:
        corr_matrix = np.corrcoef(reps.T)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
        plt.title("Neuron-Neuron Correlation Heatmap")
        plt.xlabel("Neuron Index")
        plt.ylabel("Neuron Index")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_correlation_heatmap.png"))
        plt.close()
    except Exception as e:
        print(f"Error in correlation heatmap: {e}")

    # ------------------------
    # 2️⃣ Neuron-Factor Mutual Information Heatmap
    # ------------------------
    try:
        factor_start_idx = 1 if labs.shape[1] > 5 else 0
        mi_matrix = np.zeros((n_dims, labs.shape[1] - factor_start_idx))
        for i in range(n_dims):
            for f in range(factor_start_idx, labs.shape[1]):
                dim_disc = safe_discretize(reps[:, i])
                lab_disc = labs[:, f].astype(int)
                mi_matrix[i, f - factor_start_idx] = normalized_mutual_info_score(lab_disc, dim_disc)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(mi_matrix, cmap="YlGnBu", annot=False)
        plt.title("Neuron-Factor Mutual Information")
        plt.xlabel("Ground Truth Factors")
        plt.ylabel("Latent Dimensions")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_mi_heatmap.png"))
        plt.close()
    except Exception as e:
        print(f"Error in MI heatmap: {e}")

    # ------------------------
    # 3️⃣ Graph Modularity Visualization (Neuron Graph)
    # ------------------------
    try:
        corr = np.corrcoef(reps.T)
        # Threshold weak correlations
        corr[np.abs(corr) < 0.2] = 0
        G = nx.from_numpy_array(corr)
        communities = nx.community.greedy_modularity_communities(G)
        colors = np.zeros(n_dims)
        for cid, comm in enumerate(communities):
            for node in comm:
                colors[node] = cid
        
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color=colors, cmap='tab20', node_size=80)
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        plt.title(f"Functional Modularity Graph ({len(communities)} modules)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_graph_modularity.png"))
        plt.close()
    except Exception as e:
        print(f"Error in graph modularity visualization: {e}")

    # ------------------------
    # 4️⃣ 2D Embedding Visualization (t-SNE / PCA)
    # ------------------------
    try:
        pca = PCA(n_components=2)
        pca_proj = pca.fit_transform(reps)
        tsne_proj = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42).fit_transform(reps)

        factor_idx = 1 if labs.shape[1] > 1 else 0
        colors = labs[:, factor_idx].astype(int)

        plt.figure(figsize=(8, 6))
        plt.scatter(pca_proj[:, 0], pca_proj[:, 1], c=colors, cmap='tab10', s=10, alpha=0.7)
        plt.title("PCA Projection of Representations")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_pca_projection.png"))
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=colors, cmap='tab10', s=10, alpha=0.7)
        plt.title("t-SNE Projection of Representations")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_tsne_projection.png"))
        plt.close()
    except Exception as e:
        print(f"Error in embedding visualizations: {e}")

    # ------------------------
    # 5️⃣ Activation Distribution per Neuron
    # ------------------------
    try:
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=reps, inner="box", color="skyblue", linewidth=0.5)
        plt.title("Activation Distributions per Latent Neuron")
        plt.xlabel("Neuron Index")
        plt.ylabel("Activation Value")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_activation_distribution.png"))
        plt.close()
    except Exception as e:
        print(f"Error in activation distribution plot: {e}")

    print(f"[✓] Visual modularity metrics saved to {save_dir}")

# ------------------------
# Integration with Evaluation Pipeline
# ------------------------

def evaluate_all_metrics_with_visuals(model, representations, labels, device="cpu", save_dir="results/visuals", prefix=""):
    """
    Evaluate all quantitative and visual modularity metrics.
    """
    results = evaluate_all_metrics(model, representations, labels, device=device)
    
    # Generate visual diagnostics
    visualize_functional_modularity(representations, labels, save_dir=save_dir, prefix=prefix)
    
    return results
