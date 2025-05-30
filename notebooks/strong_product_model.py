"""
Code taken from:
    https://github.com/BaileyAndrew/Strong-Product-Model/tree/main
From paper:
    https://scholar.google.co.uk/citations?view_op=view_citation&hl=en&user=N2VrHeEAAAAJ&citation_for_view=N2VrHeEAAAAJ:WF5omc3nYNoC
"""

import numpy as np
from scipy import linalg
from typing import Optional

def sim_diag(
    X: np.ndarray,
    Y: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray
]:
    """
    Simultaneously diagonalize two positive definite matrices by congruence
    (Technically only X must be positive definite, the other must be symmetric)

    Returns P, D such that X = PP^T and Y = PDP^T
    """

    X_sqrt = linalg.sqrtm(X)
    X_sqrt_inv = linalg.inv(X_sqrt)
    S = X_sqrt_inv @ Y @ X_sqrt_inv
    D, V = linalg.eigh(S)
    P = X_sqrt @ V

    return P, D

def blockwise_trace_ks(
    Lam: np.ndarray,
    D: np.ndarray
) -> np.ndarray:
    """
    Computes tr_d2[(Lam kronsum D)^-1]

    Lam, D are diagonal matrices
    """

    internal = 1 / (Lam[:, None] + D[None, :])
    return internal.sum(axis=1)

def stridewise_trace_ks(
    Lam: np.ndarray,
    D: np.ndarray
) -> np.ndarray:
    """
    Computes tr^d1[(Lam kronsum D)^-1]

    Lam, D are diagonal matrices
    """

    internal = 1 / (Lam[:, None] + D[None, :])
    return internal.sum(axis=0)

def stridewise_trace_mult(
    Lam: np.ndarray,
    D: np.ndarray
) -> np.ndarray:
    """
    Computes tr^d1[(Lam kronsum D)^-1 * (Lam kronprod I)]

    Lam, D are diagonal matrices
    """

    internal = Lam[:, None] / (Lam[:, None] * D[None, :])
    return internal.sum(axis=0)

def vec_kron_sum(Xs: list) -> np.array:
    """Compute the Kronecker vector-sum"""
    if len(Xs) == 1:
        return Xs[0]
    elif len(Xs) == 2:
        return np.kron(Xs[0], np.ones(Xs[1].shape[0])) + np.kron(np.ones(Xs[0].shape[0]), Xs[1])
    else:
        d_slash0 = np.prod([X.shape[0] for X in Xs[1:]])
        return (
            np.kron(Xs[0], np.ones(d_slash0))
            + np.kron(np.ones(Xs[0].shape[0]), vec_kron_sum(Xs[1:]))
        )

def NLL(
    Psi_1: np.ndarray,
    Theta: np.ndarray,
    Psi_2w: np.ndarray,
    S_2: np.ndarray,
    Data: np.ndarray
) -> float:
    Lam, _ = linalg.eigh(Psi_1)
    _, D = sim_diag(Theta, Psi_2w)
    _, detTheta = np.linalg.slogdet(Theta)

    if Lam.min() <= 0 or D.min() <= 0:
        # Don't allow non-positive-definite matrices
        return np.inf

    logdets = - np.log(vec_kron_sum([Lam, D])).sum() - Psi_1.shape[0] * detTheta
    traces = np.trace(Psi_2w @ S_2) + np.trace(Psi_1 @ Data @ Theta @ Data.T)

    return logdets + traces

def gradients_shifted(
    X: np.ndarray,
    S_2: np.ndarray,
    Psi_1: np.ndarray,
    V: np.ndarray,
    Lam: np.ndarray,
    Theta: np.ndarray,
    P: np.ndarray,
    Psi_2w: np.ndarray,
    D: np.ndarray,
    rho_psi_1: float,
    rho_psi_2w: float,
    rho_theta: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Computes G * Gamma, where G is the Euclidean gradient at Gamma,
    and Gamma is each of our three parameters.
    """

    P_inv = linalg.inv(P)

    Psi_1_core = blockwise_trace_ks(Lam, D) * Lam
    Psi_2w_core = stridewise_trace_ks(Lam, D) * D
    Theta_core = stridewise_trace_mult(Lam, D)

    XTheta = X @ Theta
    XtPsi = X.T @ Psi_1

    log_Psi_1 = (V * np.log(Lam)) @ V.T
    log_Psi_2w = linalg.logm(Psi_2w)
    log_Theta = linalg.logm(Theta)

    Psi_1_grad = - (V * Psi_1_core) @ V.T + XTheta @ XtPsi + 2 * rho_psi_1 * log_Psi_1
    Psi_2w_grad = - (P_inv.T * Psi_2w_core) @ P.T + S_2 @ Psi_2w + 2 * rho_psi_2w * log_Psi_2w
    Theta_grad = - (P_inv.T * Theta_core) @ P.T + XtPsi @ XTheta + 2 * rho_theta * log_Theta

    return Psi_1_grad, Theta_grad, Psi_2w_grad



def strong_product_model(
    data_matrix: np.ndarray,
    *,
    rho_rows: float,
    rho_cols_within_rows: float,
    rho_cols_between_rows: float,
    # Convergence parameters
    max_iter: int = 1000,
    tol: float = 1e-10,
    # Line search parameters
    max_iter_line_search: int = 100,
    lr_init: float = 0.1,
    lr_decay: float = 0.5,
    beta: float = 0.5,
    # Initialization
    init_rows: Optional[np.ndarray] = None,
    init_cols_between_rows: Optional[np.ndarray] = None,
    init_cols_within_rows: Optional[np.ndarray] = None,
    # Debugging tools
    verbose: bool = False,
    verbose_every: int = 1,
    return_errors: bool = False,
    # Fix parameters
    fix_Psi_1: bool = False,
    fix_Psi_2w: bool = False,
    fix_Theta: bool = False,
) -> dict[str, np.ndarray]:
    """
    Finds the graphs for the strong product model

    Parameters
    ----------
    data_matrix : np.ndarray
        The data matrix
    rho_{X}: float
        The regularization parameter for the relevant graph
    max_iter: int
        The maximum number of iterations
    tol: float
        The convergence tolerance
    max_iter_line_search: int
        The maximum number of iterations for the line search
    lr_init: float
        The initial learning rate before line search
    lr_decay: float
        The learning rate decay each line search iteration
    beta: float
        The line search parameter for the armijo rule
    init_{X}: np.ndarray
        The initial graph for X
    verbose: bool
        Whether to print out losses
    verbose_every: int
        How often to print out losses
    return_errors: bool
        Whether to return the errors
    fix_{X}: bool
        Whether to fix the graph for X at the initial graph

    Returns
    -------
    dict[str, np.ndarray]
        The dictionary containing the graphs
        Keys:
            - 'rows': The graph for the rows
            - 'cols_between_rows': The graph for the columns between rows
            - 'cols_within_rows': The graph for the columns within rows

        Note that in the paper and code, we use Theta to denote cols_between_rows + I
        since it is a more mathematically convenient representation.  We do not return
        Theta, though; we remove I before returning.

    """

    d_1, d_2 = data_matrix.shape

    if return_errors:
        errors = []

    # The identity matrix makes a good default initialization
    if init_rows is None:
        init_rows = np.eye(d_1)
    if init_cols_between_rows is None:
        init_cols_between_rows = np.eye(d_2)
    if init_cols_within_rows is None:
        init_cols_within_rows = np.eye(d_2)

    # Initialize the variables
    Psi_1 = init_rows
    Psi_2w = init_cols_within_rows
    Theta = init_cols_between_rows + np.eye(d_2)

    # Useful precomputation
    S_2 = data_matrix.T @ data_matrix

    old_NLL = NLL(Psi_1, Theta, Psi_2w, S_2, data_matrix)
    if verbose:
        print(f"Iteration 0: {old_NLL}")
    if return_errors:
        errors.append(old_NLL)

    for i in range(max_iter):
        # Store starting point
        old_Psi_1 = Psi_1
        old_Psi_2w = Psi_2w
        old_Theta = Theta

        # Eigendecompose the first matrix
        Lam, V = linalg.eigh(Psi_1)

        # Simultaneously diagonalize the other two
        P, D = sim_diag(Theta, Psi_2w)

        # Get grad^S * X, an intermediate step
        # between gradient and retraction
        A, C, B = gradients_shifted(
            data_matrix,
            S_2,
            Psi_1,
            V,
            Lam,
            Theta,#Psi_2w,
            P,
            Psi_2w,#Theta,
            D,
            rho_rows,
            rho_cols_within_rows,
            rho_cols_between_rows,
        )

        # Get new points
        lr = lr_init
        if not fix_Psi_1:
            Psi_1 = old_Psi_1 @ linalg.expm(-lr * A)
        if not fix_Psi_2w:
            Psi_2w = old_Psi_2w @ linalg.expm(-lr * B)
        if not fix_Theta:
            Theta = old_Theta @ linalg.expm(-lr * C)

        # Measure the size of the gradient, for Armijo
        grad_norm = (
            np.trace(np.linalg.matrix_power(A @ np.linalg.inv(old_Psi_1), 2))
            + np.trace(np.linalg.matrix_power(B @ np.linalg.inv(old_Psi_2w), 2))
            + np.trace(np.linalg.matrix_power(C @ np.linalg.inv(old_Theta), 2))
        )

        # Line search
        try:
            new_NLL = NLL(Psi_1, Theta, Psi_2w, S_2, data_matrix)
        except:
            new_NLL = np.inf

        converged = False
        for _ in range(max_iter_line_search):
            if new_NLL < old_NLL - lr * beta * grad_norm:
                # Check if all matrices are positive definite
                Psi_1_posdef = (np.linalg.eigh(Psi_1)[0] > 0).all()
                Psi_2w_posdef = (np.linalg.eigh(Psi_2w)[0] > 0).all()
                Theta_posdef = (np.linalg.eigh(Theta)[0] > 1).all()
                if Psi_1_posdef and Psi_2w_posdef and Theta_posdef:
                    break
            lr *= lr_decay
            if not fix_Psi_1:
                Psi_1 = old_Psi_1 @ linalg.expm(-lr * A)
            if not fix_Psi_2w:
                Psi_2w = old_Psi_2w @ linalg.expm(-lr * B)
            if not fix_Theta:
                Theta = old_Theta @ linalg.expm(-lr * C)
            try:
                new_NLL = NLL(Psi_1, Theta, Psi_2w, S_2, data_matrix)
            except:
                new_NLL = np.inf
        else:
            # If we never break, were unable to improve the loss,
            # and hence we converged!
            Psi_1 = old_Psi_1
            Psi_2w = old_Psi_2w
            Theta = old_Theta
            new_NLL = old_NLL
            converged = True

        if abs(new_NLL - old_NLL) < tol:
            converged = True
        
        if return_errors:
            errors.append(new_NLL)

        if converged:
            if verbose:
                print(f"Iteration {i+1}: {new_NLL} (converged)")
            break
        else:
            if verbose and i % verbose_every == 0:
                print(f"Iteration {i+1}: {new_NLL}")

        old_NLL = new_NLL

    
    to_return = {
        "rows": Psi_1,
        "cols_within_rows": Psi_2w,
        "cols_between_rows": Theta - np.eye(d_2),
    }

    if return_errors:
        to_return["errors"] = errors

    return to_return

    