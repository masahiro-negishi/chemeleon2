"""Distance matrix-based decoder for VAE.

Predicts pairwise distance matrices instead of fractional coordinates.
This aligns better with MACE embeddings which encode relative geometry.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch

from src.utils.scatter import scatter_mean


class DistanceMatrixDecoder(nn.Module):
    """Transformer decoder that predicts pairwise distance matrices.

    Instead of predicting fractional coordinates, this decoder predicts
    symmetric, non-negative distance matrices with zero diagonal.
    This is more aligned with MACE embeddings which encode relative geometry.
    """

    def __init__(
        self,
        atom_type_predict: bool = True,
        max_num_elements: int = 100,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_first: bool = True,
        bias: bool = True,
        num_layers: int = 8,
        distance_mlp_hidden_dims: list[int] | None = None,
        classifier_hidden_dims: list[int] | None = None,
        use_vectorized: bool = True,
    ) -> None:
        """Initialize distance matrix decoder.

        Uses permutation-invariant pairwise features [h_i + h_j || h_i * h_j] where
        both addition and multiplication are commutative operations, ensuring the MLP
        receives identical features for pairs (i,j) and (j,i).

        Args:
            atom_type_predict: Whether to predict atom types
            max_num_elements: Maximum number of element types
            d_model: Transformer hidden dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward dimension
            activation: Activation function ("gelu" or "relu")
            dropout: Dropout rate
            norm_first: Whether to apply layer norm before attention
            bias: Whether to use bias in linear layers
            num_layers: Number of transformer layers
            distance_mlp_hidden_dims: Hidden dimensions for distance prediction MLP
            classifier_hidden_dims: Hidden dimensions for classifier MLP
            use_vectorized: Whether to use vectorized distance matrix computation
        """
        super().__init__()

        self.max_num_elements = max_num_elements
        self.d_model = d_model
        self.num_layers = num_layers
        self.atom_type_predict = atom_type_predict
        self.use_vectorized = use_vectorized

        if distance_mlp_hidden_dims is None:
            distance_mlp_hidden_dims = [d_model, d_model // 2]

        if classifier_hidden_dims is None:
            classifier_hidden_dims = [d_model, d_model // 2]

        # Store activation type for creating multiple instances
        activation_map = {
            "gelu": lambda: nn.GELU(approximate="tanh"),
            "relu": lambda: nn.ReLU(),
        }
        activation_fn = activation_map[activation]()

        # Transformer encoder for processing latent features
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation_fn,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
                bias=bias,
            ),
            norm=nn.LayerNorm(d_model),
            num_layers=num_layers,
        )

        # Prediction heads
        if atom_type_predict:
            self.atom_types_head = nn.Linear(d_model, max_num_elements, bias=True)

        self.lattice_head = nn.Linear(d_model, 6, bias=False)

        # Distance matrix prediction head
        # Input: permutation-invariant features (sum + product)
        # [h_i + h_j || h_i * h_j] -> MLP -> scalar distance
        distance_mlp_input_dim = 2 * d_model
        layers = []
        prev_dim = distance_mlp_input_dim
        for hidden_dim in distance_mlp_hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    activation_map[activation](),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # Output: scalar distance
        self.distance_mlp = nn.Sequential(*layers)

        # Classifier MLP: [h_i + h_j || h_i * h_j] -> scalar logit
        classifier_input_dim = 2 * d_model  # Same as distance_mlp
        layers = []
        prev_dim = classifier_input_dim
        for hidden_dim in classifier_hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    activation_map[activation](),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # Output: scalar logit
        self.classifier_mlp = nn.Sequential(*layers)

    @property
    def hidden_dim(self) -> int:
        return self.d_model

    def forward(
        self, encoded_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Forward pass of distance matrix decoder.

        Args:
            encoded_batch: Dict with the following attributes:
                x (torch.Tensor): Encoded batch of atomic environments (num_nodes, d_model)
                num_atoms (torch.Tensor): Number of atoms in each structure
                batch (torch.Tensor): Batch index for each atom
                token_idx (torch.Tensor): Token index for each atom

        Returns:
            Dict with:
                atom_types: (num_nodes, max_num_elements) if atom_type_predict else None
                lattices: (num_graphs, 6) lattice parameters
                lengths: (num_graphs, 3) lattice lengths
                angles: (num_graphs, 3) lattice angles
                distance_matrix: List of (N_i, N_i) distance matrices
                distance_classifier_logits: List of (N_i, N_i) classifier logits
                frac_coords: Dummy tensor of zeros (num_nodes, 3)
        """
        x = encoded_batch["x"]
        batch_idx = encoded_batch["batch"]
        num_atoms = encoded_batch["num_atoms"]

        # Convert from PyG batch to dense batch with padding
        x, token_mask = to_dense_batch(x, batch_idx)

        # Transformer forward pass
        x = self.transformer.forward(x, src_key_padding_mask=(~token_mask))
        x = x[token_mask]  # (num_nodes, d_model)

        # Global pooling: (num_nodes, d_model) -> (num_graphs, d_model)
        x_global = scatter_mean(x, batch_idx, dim=0)

        # Atomic type prediction head
        if self.atom_type_predict:
            atom_types_out = self.atom_types_head(x)
        else:
            atom_types_out = None

        # Lattice lengths and angles prediction head
        lattices_out = self.lattice_head(x_global)

        # Distance matrix prediction
        distance_matrices, classifier_logits = self._predict_distance_matrices(
            x, batch_idx, num_atoms
        )

        # Dummy fractional coordinates (for interface compatibility)
        frac_coords_out = torch.zeros((x.shape[0], 3), device=x.device)

        result = {
            "atom_types": atom_types_out,
            "lattices": lattices_out,
            "lengths": lattices_out[:, :3],
            "angles": lattices_out[:, 3:],
            "distance_matrix": distance_matrices,
            "distance_classifier_logits": classifier_logits,
            "frac_coords": frac_coords_out,
        }
        return result

    def _predict_distance_matrices(
        self,
        x: torch.Tensor,
        batch_idx: torch.Tensor,
        num_atoms: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Predict distance matrices with automatic fallback for large batches.

        Attempts vectorized computation for efficiency. Falls back to sequential
        processing if the batch is too large (>100K pairs) or if OOM is detected.

        Args:
            x: Per-atom features (num_nodes, d_model)
            batch_idx: Batch index for each atom (num_nodes,)
            num_atoms: Number of atoms per structure (num_graphs,)

        Returns:
            Tuple of (distance_matrices, classifier_logits_list)
        """
        if not self.use_vectorized:
            return self._predict_distance_matrices_sequential(x, batch_idx, num_atoms)

        # Memory safety check
        num_pairs_per_graph = num_atoms * (num_atoms + 1) // 2
        total_pairs = num_pairs_per_graph.sum().item()

        # Conservative threshold: 100K pairs ≈ 400 MB
        MAX_PAIRS_THRESHOLD = 100_000

        if total_pairs > MAX_PAIRS_THRESHOLD:
            # Fallback to sequential for safety
            return self._predict_distance_matrices_sequential(x, batch_idx, num_atoms)

        try:
            return self._predict_distance_matrices_vectorized(x, batch_idx, num_atoms)
        except RuntimeError as e:
            if "out of memory" in str(e):
                # OOM detected, clear cache and fallback
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self._predict_distance_matrices_sequential(
                    x, batch_idx, num_atoms
                )
            raise

    def _predict_distance_matrices_sequential(
        self,
        x: torch.Tensor,
        batch_idx: torch.Tensor,
        num_atoms: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Sequential processing (fallback for large batches).

        Computes permutation-invariant features [h_i + h_j || h_i * h_j] for each
        unique atom pair (i,j) where i <= j, then mirrors to lower triangle for
        symmetric output.

        Args:
            x: Per-atom features (num_nodes, d_model)
            batch_idx: Batch index for each atom (num_nodes,)
            num_atoms: Number of atoms per structure (num_graphs,)

        Returns:
            Tuple of (distance_matrices, classifier_logits_list)
        """
        distance_matrices = []
        classifier_logits_list = []
        num_graphs = len(num_atoms)

        node_offset = 0
        for graph_idx in range(num_graphs):
            n_atoms = num_atoms[graph_idx].item()
            h = x[node_offset : node_offset + n_atoms]  # (N_i, d_model)

            # Extract upper triangle indices (excluding diagonal)
            triu_indices = torch.triu_indices(
                n_atoms, n_atoms, offset=1, device=h.device
            )
            i_indices = triu_indices[0]  # (num_pairs,)
            j_indices = triu_indices[1]  # (num_pairs,)

            # Extract features for pairs
            h_i_pairs = h[i_indices]  # (num_pairs, d_model)
            h_j_pairs = h[j_indices]  # (num_pairs, d_model)

            # Compute permutation-invariant features
            pairwise_features = torch.cat(
                [
                    h_i_pairs + h_j_pairs,  # (num_pairs, d_model)
                    h_i_pairs * h_j_pairs,  # (num_pairs, d_model)
                ],
                dim=-1,
            )  # (num_pairs, 2*d_model)

            # Predict distances for upper triangle
            distances_upper = self.distance_mlp(pairwise_features).squeeze(
                -1
            )  # (num_pairs,)
            distances_upper = F.softplus(distances_upper)

            # Predict classifier logits for upper triangle
            logits_upper = self.classifier_mlp(pairwise_features).squeeze(
                -1
            )  # (num_pairs,)

            # Build symmetric distance matrix
            distances = torch.zeros(n_atoms, n_atoms, device=h.device)
            distances[i_indices, j_indices] = distances_upper
            distances = (
                distances + distances.T
            )  # Mirror to lower triangle, diagonal stays 0

            # Build symmetric classifier logits matrix
            logits = torch.zeros(n_atoms, n_atoms, device=h.device)
            logits[i_indices, j_indices] = logits_upper
            logits = logits + logits.T  # Mirror to lower triangle, diagonal stays 0

            distance_matrices.append(distances)
            classifier_logits_list.append(logits)
            node_offset += n_atoms

        return distance_matrices, classifier_logits_list

    def _predict_distance_matrices_vectorized(
        self,
        x: torch.Tensor,
        batch_idx: torch.Tensor,
        num_atoms: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Vectorized distance matrix prediction via concatenation.

        Processes all atom pairs across all graphs in a single batched MLP forward pass.
        This eliminates the sequential loop and maximizes GPU utilization.

        Algorithm:
        1. Compute pair counts and offsets for all graphs
        2. Build global pair indices for all graphs
        3. Concatenate pairwise features: [h_i + h_j || h_i * h_j]
        4. Single MLP forward pass for all pairs
        5. Split output and reconstruct per-graph matrices

        Args:
            x: Per-atom features (num_nodes, d_model)
            batch_idx: Batch index for each atom (num_nodes,)
            num_atoms: Number of atoms per structure (num_graphs,)

        Returns:
            Tuple of (distance_matrices, classifier_logits_list)
        """
        num_graphs = len(num_atoms)

        # 1. Compute offsets
        num_pairs_per_graph = (
            num_atoms * (num_atoms - 1) // 2
        )  # Upper triangle excluding diagonal
        pair_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=x.device),
                torch.cumsum(num_pairs_per_graph, dim=0),
            ]
        )
        node_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=x.device),
                torch.cumsum(num_atoms, dim=0)[:-1],
            ]
        )
        total_pairs = pair_offsets[-1].item()

        # 2. Build global pair indices
        all_i_indices = torch.empty(total_pairs, dtype=torch.long, device=x.device)
        all_j_indices = torch.empty(total_pairs, dtype=torch.long, device=x.device)

        for graph_idx in range(num_graphs):
            n_atoms = num_atoms[graph_idx].item()
            node_offset = node_offsets[graph_idx].item()
            pair_start = pair_offsets[graph_idx].item()
            pair_end = pair_offsets[graph_idx + 1].item()

            triu_indices = torch.triu_indices(
                n_atoms, n_atoms, offset=1, device=x.device
            )
            all_i_indices[pair_start:pair_end] = triu_indices[0] + node_offset
            all_j_indices[pair_start:pair_end] = triu_indices[1] + node_offset

        # 3. Extract and concatenate pairwise features
        h_i_all = x[all_i_indices]  # (total_pairs, d_model)
        h_j_all = x[all_j_indices]  # (total_pairs, d_model)

        pairwise_features = torch.cat(
            [
                h_i_all + h_j_all,  # Commutative sum
                h_i_all * h_j_all,  # Commutative product
            ],
            dim=-1,
        )  # (total_pairs, 2*d_model)

        # 4. Parallel MLP forward passes
        distances_all = self.distance_mlp(pairwise_features).squeeze(-1)
        distances_all = F.softplus(distances_all)  # Non-negativity

        classifier_logits_all = self.classifier_mlp(pairwise_features).squeeze(-1)

        # 5. Reconstruct per-graph matrices for both outputs
        distance_matrices = []
        classifier_logits_list = []

        for graph_idx in range(num_graphs):
            n_atoms = num_atoms[graph_idx].item()
            pair_start = pair_offsets[graph_idx].item()
            pair_end = pair_offsets[graph_idx + 1].item()

            distances_upper = distances_all[pair_start:pair_end]
            logits_upper = classifier_logits_all[pair_start:pair_end]

            # Build symmetric distance matrix
            triu_indices = torch.triu_indices(
                n_atoms, n_atoms, offset=1, device=x.device
            )
            distances = torch.zeros(n_atoms, n_atoms, device=x.device)
            distances[triu_indices[0], triu_indices[1]] = distances_upper
            distances = (
                distances + distances.T
            )  # Mirror to lower triangle, diagonal stays 0

            # Build symmetric classifier logits matrix
            logits = torch.zeros(n_atoms, n_atoms, device=x.device)
            logits[triu_indices[0], triu_indices[1]] = logits_upper
            logits = logits + logits.T  # Mirror to lower triangle, diagonal stays 0

            distance_matrices.append(distances)
            classifier_logits_list.append(logits)

        return distance_matrices, classifier_logits_list
