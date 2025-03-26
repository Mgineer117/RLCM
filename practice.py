import torch


def compute_samplewise_ABK(x: torch.Tensor, dot_x: torch.Tensor) -> torch.Tensor:
    """
    Compute a local estimate of the Jacobian (A+BK) at each sample by performing
    a local regression using neighboring states.

    Parameters:
      x: Tensor of shape (T, n) representing T time steps (or samples) of the state.
      dot_x: Tensor of shape (T, n) representing the corresponding time derivative
             approximations at each sample.

    Returns:
      ABK_local: Tensor of shape (T, n, n) where each slice ABK_local[i] is the estimated
                 Jacobian at sample i.
    """
    T, n = x.shape
    ABK_local = torch.zeros(T, n, n, device=x.device, dtype=x.dtype)

    for i in range(T):
        # Collect indices of neighboring samples
        neighbor_indices = []
        if i - 1 >= 0:
            neighbor_indices.append(i - 1)
        if i + 1 < T:
            neighbor_indices.append(i + 1)

        # If we don't have any neighbors, we cannot estimate a local Jacobian
        if len(neighbor_indices) == 0:
            ABK_local[i] = torch.zeros(n, n, device=x.device, dtype=x.dtype)
            continue

        # Form the local differences relative to sample i.
        # We use differences both in state and in the approximate derivative.
        local_delta_x = []
        local_delta_dot = []
        for j in neighbor_indices:
            local_delta_x.append((x[j] - x[i]).unsqueeze(0))  # shape (1, n)
            local_delta_dot.append((dot_x[j] - dot_x[i]).unsqueeze(0))  # shape (1, n)

        # Stack the local differences: X_local shape (k, n) and Y_local shape (k, n),
        # where k is the number of neighbors (typically 2).
        X_local = torch.cat(local_delta_x, dim=0)
        Y_local = torch.cat(local_delta_dot, dim=0)

        # Solve for the local Jacobian J_i such that: X_local @ J_i ≈ Y_local.
        # If there are enough neighbors (k >= n), we use lstsq. Otherwise, we fall back to a pseudo-inverse.
        if X_local.shape[0] >= n:
            J_local, res, rank, s = torch.linalg.lstsq(X_local, Y_local, rcond=None)
        else:
            J_local = torch.linalg.pinv(X_local) @ Y_local

        ABK_local[i] = J_local

    return ABK_local


# Example usage:
if __name__ == "__main__":
    # Generate a synthetic trajectory:
    T, n = 50, 4
    dt = 0.01
    # Suppose x follows some dynamics; here we'll simulate with a linear system for example.
    true_J = torch.randn(n, n)
    x = torch.zeros(T, n)
    x[0] = torch.randn(n)
    for i in range(1, T):
        x[i] = x[i - 1] + dt * (x[i - 1] @ true_J.T)

    # Compute dot_x using a simple first-order forward difference.
    dot_x = torch.zeros_like(x)
    dot_x[:-1] = (x[1:] - x[:-1]) / dt
    dot_x[-1] = dot_x[-2]  # approximate last derivative

    # Estimate sample-wise ABK:
    ABK_local = compute_samplewise_ABK(x, dot_x)
    print("Estimated sample-wise A+BK shape:", ABK_local.shape)
