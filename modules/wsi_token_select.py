import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

class CrossAttentionTokenReducer(nn.Module):
    def __init__(self, hidden_dim, target_length, num_heads):
        """
        Initializes the cross-attention token reducer layer.

        Args:
            hidden_dim (int): The dimensionality of the input and output features.
            target_length (int): The desired sequence length after reduction.
            num_heads (int): The number of attention heads.
        """
        super(CrossAttentionTokenReducer, self).__init__()
        self.hidden_dim = hidden_dim
        self.target_length = target_length
        self.num_heads = num_heads

        # Initialize query as a learnable parameter
        self.query = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(1, target_length, hidden_dim), 0., 0.2))

        # Multi-head attention layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

    def forward(self, x):
        """
        Forward pass for the token reduction layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_length, hidden_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, target_length, hidden_dim).
        """
        # Reshape input to match the shape expected by nn.MultiheadAttention
        # Shape becomes (seq_length, batch_size, hidden_dim)
        x = x.permute(1, 0, 2)

        # Repeat the query for each item in the batch and permute
        query = self.query.repeat(x.size(1), 1, 1).permute(1, 0, 2)

        # Apply multi-head attention
        # Output shape: (target_length, batch_size, hidden_dim)
        attn_output, _ = self.multihead_attn(query=query, key=x, value=x)

        # Reshape output to (batch_size, target_length, hidden_dim)
        attn_output = attn_output.permute(1, 0, 2)

        return attn_output
    
def uniform_sampling_batch(features, desired_length):
    """
    Reduces the sequence length of the features to a desired length using uniform sampling for batch processing.

    Args:
        features (torch.Tensor): The input features with shape [batch_size, seq_length, feature_dim].
        desired_length (int): The desired sequence length after reduction.

    Returns:
        torch.Tensor: The output features with shape [batch_size, desired_length, feature_dim].
    """
    batch_size, seq_length, feature_dim = features.size()
    if seq_length <= desired_length:
        return features

    indices = torch.linspace(0, seq_length - 1, steps=desired_length).long()
    sampled_features = features[:, indices, :]

    return sampled_features

# def kmeans_reduction_batch(features, desired_length):
#     """
#     Reduces the sequence length of the features to a desired length using k-means clustering for batch processing.

#     Args:
#         features (torch.Tensor): The input features with shape [batch_size, seq_length, feature_dim].
#         desired_length (int): The desired sequence length after reduction.

#     Returns:
#         torch.Tensor: The output features with shape [batch_size, desired_length, feature_dim].
#     """
#     batch_size, seq_length, feature_dim = features.size()
#     reduced_features = []
    
#     for i in range(batch_size):
#         single_feature = features[i]
#         if seq_length <= desired_length:
#             reduced_features.append(single_feature)
#             continue

#         feature_array = single_feature.cpu().detach().numpy()
#         kmeans = KMeans(n_clusters=desired_length, random_state=0).fit(feature_array)
#         centroids = kmeans.cluster_centers_
#         centroids_tensor = torch.tensor(centroids, dtype=torch.float32).to(features.device)
#         reduced_features.append(centroids_tensor)

#     reduced_features = torch.stack(reduced_features)

#     return reduced_features

def kmeans_reduction_batch(features, desired_length, use_pca=True, pca_components=512):
    """
    Reduces the sequence length of the features to a desired length using MiniBatch K-means clustering for batch processing.

    Args:
        features (torch.Tensor): Input features with shape [batch_size, seq_length, feature_dim].
        desired_length (int): Desired sequence length after reduction.
        use_pca (bool): Whether to apply PCA for dimensionality reduction before K-Means.
        pca_components (int): Number of components for PCA. If None, it will not reduce dimensions.

    Returns:
        torch.Tensor: Output features with shape [batch_size, desired_length, feature_dim].
    """
    batch_size, seq_length, feature_dim = features.size()
    reduced_features = []

    # Convert to numpy once to avoid repeated data transfer
    features_np = features.cpu().detach().numpy()

    for i in range(batch_size):
        single_feature = features_np[i]
        
        # Skip reduction if the sequence is already shorter than the desired length
        if seq_length <= desired_length:
            reduced_features.append(torch.tensor(single_feature, dtype=torch.float32).to(features.device))
            continue

        # Apply PCA if enabled
        if use_pca and pca_components is not None:
            pca = PCA(n_components=pca_components)
            single_feature = pca.fit_transform(single_feature)

        # Apply MiniBatch KMeans
        kmeans = MiniBatchKMeans(n_clusters=desired_length, random_state=0)
        kmeans.fit(single_feature)
        centroids = kmeans.cluster_centers_
        centroids_tensor = torch.tensor(centroids, dtype=torch.float32).to(features.device)
        reduced_features.append(centroids_tensor)

    return torch.stack(reduced_features)