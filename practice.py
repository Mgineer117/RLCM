import torch


def compute_B_perp_batch(BK, u_dim, method="svd", threshold=1e-5):
    """
    Compute the nullspace basis B_perp for each sample in the batch from BK,
    using either SVD or QR decomposition, and return a tensor of shape
    (batch, x_dim, u_dim). For each sample, the columns of B_perp correspond
    to an orthonormal basis of the nullspace of BK, padded (with zeros) or truncated
    to have u_dim columns.

    Parameters:
      BK: Tensor of shape (batch, x_dim, x_dim), the estimated BK for each sample.
      u_dim: The control dimension, i.e. the desired number of nullspace vectors.
      method: String, either "svd" or "qr" to select the decomposition method.
      threshold: Threshold below which singular (or R diagonal) values are considered zero.

    Returns:
      B_perp_tensor: Tensor of shape (batch, x_dim, u_dim).
    """
    batch_size, x_dim, _ = BK.shape
    B_perp_list = []

    for i in range(batch_size):
        BK_i = BK[i]  # shape: (x_dim, x_dim)
        if method.lower() == "svd":
            # Use SVD: BK_i = U Sigma V^T.
            U, S, _ = torch.linalg.svd(BK_i)
            # Nullspace: columns of U corresponding to singular values < threshold.
            null_indices = (S < threshold).nonzero(as_tuple=True)[0]
            if null_indices.numel() > 0:
                B_perp_i = U[
                    :, null_indices
                ]  # shape: (x_dim, m) where m is number of null directions.
            else:
                B_perp_i = torch.empty(x_dim, 0, device=BK.device, dtype=BK.dtype)
        elif method.lower() == "qr":
            # Use QR on the transpose: compute QR of BK_i^T = Q R.
            Q, R = torch.linalg.qr(BK_i.T)  # Q: (x_dim, x_dim), R: (x_dim, x_dim)
            # Check the absolute values of the diagonal of R.
            diag_R = torch.abs(torch.diag(R))
            null_indices = (diag_R < threshold).nonzero(as_tuple=True)[0]
            if null_indices.numel() > 0:
                B_perp_i = Q[
                    :, null_indices
                ]  # Q's columns corresponding to near-zero diag elements.
            else:
                B_perp_i = torch.empty(x_dim, 0, device=BK.device, dtype=BK.dtype)
        else:
            raise ValueError("Method must be either 'svd' or 'qr'.")

        # Now, B_perp_i is of shape (x_dim, m). We want output of shape (x_dim, u_dim).
        # Create a zero matrix of shape (x_dim, u_dim) and fill in as many columns as available.
        padded = torch.zeros(x_dim, u_dim, device=BK.device, dtype=BK.dtype)
        m = B_perp_i.shape[1]
        if m > 0:
            if m >= u_dim:
                padded[:, :] = B_perp_i[:, :u_dim]
            else:
                padded[:, :m] = B_perp_i
        B_perp_list.append(padded)

    B_perp_tensor = torch.stack(B_perp_list, dim=0)  # shape: (batch, x_dim, u_dim)
    return B_perp_tensor


# Example usage:
batch_size = 5
x_dim = 6
u_dim = 2
BK = torch.randn(batch_size, x_dim, x_dim)
B_perp_svd = compute_B_perp_batch(BK, u_dim, method="svd", threshold=1e-5)
B_perp_qr = compute_B_perp_batch(BK, u_dim, method="qr", threshold=1e-5)
print("B_perp (SVD) shape:", B_perp_svd.shape)
print("B_perp (QR) shape:", B_perp_qr.shape)
print("Error:", torch.linalg.matrix_norm(B_perp_svd - B_perp_qr).mean())
