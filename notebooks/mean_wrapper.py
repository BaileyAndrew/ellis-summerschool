"""
Code from:
    https://github.com/BaileyAndrew/Noncentral-KS-Normal/tree/master
From paper:
    https://arxiv.org/abs/2408.02393

"""

import numpy as np
import scipy.sparse as sparse
from GmGM import Dataset
from GmGM.typing import Axis
from typing import Callable, Literal

class NoncentralKS:
    """
    Used to estimate the mean and precisions of a noncentral
    Kronecker-sum-structured distribution.

    Wraps a central Kronecker-sum-structured distribution parameter estimator

    Note that the implementation could be improved to use rank-one updates of
    the sufficient statistics (Gram matrices), and even could update the
    eigenvectors directly if this code were made more tightly intertwined with
    a central KS estimator.  However, this implementation is simpler and can
    be used with any estimator, hence our preference for implementing our
    paper's experiments with this version.

    Parameters:
    -----------
    estimator : callable[Dataset -> dict[Axis, np.ndarray]]
        An object that implements the `fit` method, which takes in a data matrix
        and returns the estimated mean and precision of the distribution.

        Output is dictionary of precision matrices
    initial_mean : tuple[dict[Axis, np.ndarray], float]
        The initial mean estimate of the distribution.
    initial_precision : dict[Axis, np.ndarray]
        The initial precision estimate of the distribution.
    """

    def __init__(
        self,
        estimator: Callable[[Dataset], dict[Axis, np.ndarray]],
        initial_mean: tuple[dict[Axis, np.ndarray], float],
        initial_precision: dict[Axis, np.ndarray],
    ) -> None:
        self.estimator = estimator
        self.column_means = initial_mean[0]
        self.full_mean = initial_mean[1]
        self.precision = initial_precision

    def fit(
        self,
        data: Dataset,
        max_iter: int = 100,
        tol: float = 1e-4,
        converge_by: Literal[
            "means",
            "precisions",
            "objective",
            "pseudo-objective"
        ] = "precisions",
        verbose: bool = False,
    ) -> tuple[
        tuple[dict[Axis, np.ndarray], float],
        dict[Axis, np.ndarray]
    ]:
        """
        Estimate the mean and precision of the distribution.

        Parameters:
        -----------
        data : Dataset
            The data matrix to estimate the distribution from.

        Returns:
        --------
        dict[Axis, np.ndarray]
            The estimated mean and precision of the distribution.
        """
        
        if len(data.dataset) != 1:
            raise ValueError("NoncentralKS only supports one dataset at a time")
        key = list(data.dataset.keys())[0]

        orig_data = data.dataset[key].copy()
        indices = {
            ell: data.structure[key].index(ell)
            for ell in data.structure[key]
            if ell not in data.batch_axes
        }

        converged = False
        num_iters = 0
        prev_means = (self.column_means.copy(), self.full_mean)
        prev_precisions = self.precision.copy()
        prev_objective = np.inf

        while not converged:
            # Remove the mean estimate from the data
            data.dataset = {key: orig_data - self.full_mean}
            for axis in data.all_axes - data.batch_axes:
                data.dataset = {
                    key: data.dataset[key] - self.column_means[axis].reshape(
                        *(
                            [1] * indices[axis]
                            + [-1]
                            + [1] * (orig_data.ndim - indices[axis] - 1)
                        )
                    )
                }

            # Update precision estimates
            self.precision = self.estimator(data)

            # Update the mean estimates
            self.column_means, self.full_mean = mean_estimator(
                orig_data,
                self.precision,
                (self.column_means, self.full_mean),
                np.prod(orig_data.shape),
                [axis for axis in data.structure[key]],
                data.batch_axes
            )

            # Check if we should stop
            num_iters += 1
            if num_iters >= max_iter:
                if verbose:
                    print("Maximum number of iterations reached")
                converged = True

            # Check for convergence
            if converge_by == "means":
                squared_dist_1 = sum(
                    ((prev_means[0][axis] - self.column_means[axis])**2).sum()
                    for axis in data.structure[key]
                    if axis not in data.batch_axes
                )
                squared_dist_2 = (prev_means[1] - self.full_mean)**2
                dist = np.sqrt(squared_dist_1 + squared_dist_2)

                if dist < tol:
                    converged = True
                    if verbose:
                        print(f"Converged in {num_iters} iterations")
            elif converge_by == "precisions":
                dist = sum(
                    ((prev_precisions[axis] - self.precision[axis])**2).sum()
                    for axis in data.structure[key]
                    if axis not in data.batch_axes
                )
                dist = np.sqrt(dist)

                if dist < tol:
                    converged = True
                    if verbose:
                        print(f"Converged in {num_iters} iterations")
            elif converge_by == "objective":
                # Not practically feasible (involves very large logdet)
                objective = self.objective(orig_data, data.structure[key])
                dist = np.abs(prev_objective - objective)
                prev_objective = objective

                if dist < tol:
                    converged = True
                    if verbose:
                        print(f"Converged in {num_iters} iterations")
            elif converge_by == "pseudo-objective":
                pseudo_objective = self.pseudo_objective(
                    orig_data,
                    data.structure[key]
                )
                dist = np.abs(prev_objective - pseudo_objective)
                prev_objective = pseudo_objective

                if dist < tol:
                    converged = True
                    if verbose:
                        print(f"Converged in {num_iters} iterations")

            if verbose:
                print(f"Iteration: {num_iters} (Change: {dist})")

            # Update the previous mean estimate
            prev_means = (self.column_means.copy(), self.full_mean)
            prev_precisions = self.precision.copy()

        # Reset the data to its original form
        data.dataset[key] = orig_data

        return (self.column_means, self.full_mean), self.precision
    
    def objective(self, orig_data: np.ndarray, axes: list) -> float:
        # Infeasible runtime
        centralized = orig_data.reshape(-1) - vec_kron_sum([
            self.column_means[axis]
            for axis in axes
        ]) - self.full_mean
        matrix = kron_sum([
            self.precision[axis]
            for axis in axes
        ])
        return sparse_logdet(matrix) - centralized @ matrix @ centralized
    
    def pseudo_objective(self, orig_data: np.ndarray, axes: list) -> float:
        centralized = orig_data.reshape(-1) - vec_kron_sum([
            self.column_means[axis]
            for axis in axes
        ]) - self.full_mean
        matrix = kron_sum([
            self.precision[axis]
            for axis in axes
        ])
        return centralized @ matrix @ centralized
    
def sparse_logdet(matrix: sparse.spmatrix) -> float:
    """Compute the log-determinant of a sparse matrix"""
    lu = sparse.linalg.splu(matrix)
    logdet = np.log(lu.U.diagonal()).sum() + np.log(lu.L.diagonal()).sum()
    return logdet
    
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
    
def kron_sum(Xs: list) -> np.array:
    """Compute the Kronecker sum"""
    if len(Xs) == 1:
        return Xs[0]
    elif len(Xs) == 2:
        return sparse.kronsum(Xs[0], Xs[1])
    else:
        return sparse.kronsum(Xs[0], kron_sum(Xs[1:]))

def mean_estimator(
    data: np.ndarray,
    Psis: dict[Axis, np.ndarray],
    initial_mean: tuple[dict[Axis, np.ndarray], float],
    d_full: float,
    axes: list[Axis],
    batch_axes: list[Axis]
) -> tuple[dict[Axis, np.ndarray], float]:
    
    # Remove the batch axes from the axes
    batch_axes = [batch_axis for batch_axis in batch_axes if batch_axis in axes]
    data = data.sum(axis=tuple([
        axes.index(batch_axis)
        for batch_axis in batch_axes
    ]))
    for batch_axis in batch_axes:
        index_of_batch = axes.index(batch_axis)
        axes = axes[:index_of_batch] + axes[index_of_batch+1:]


    # Derived parameters for our mean problem
    means = initial_mean[0]
    full_mean = initial_mean[1]
    lsum_Psis = {ell: Psis[ell].sum(axis=1) for ell in axes}
    sum_Psis = {ell: lsum_Psis[ell].sum() for ell in axes}
    ds = {ell: Psis[ell].shape[0] for ell in axes}
    d_slashes = {ell: d_full / ds[ell] for ell in axes}
    sum_Psis_slashes = {
        ell_prime: sum([
            d_slashes[ell] / ds[ell_prime] * sum_Psis[ell]
            for ell in axes if ell != ell_prime
        ])
        for ell_prime in axes
    }
    indices_dict = {ell: axes.index(ell) for ell in axes}

    # The matrix that needs to be inverted
    A = {
        ell: (
            d_slashes[ell] * Psis[ell]
            + sum_Psis_slashes[ell] * np.eye(ds[ell])
        )
        for ell in axes
    }
    A_inv = {ell: np.linalg.pinv(A[ell]) for ell in axes}

    # The data contribution
    def datatrans(ell, data, Psis):
        # Sum along all axes but ell
        base = data.sum(axis=tuple([
            indices_dict[ell_prime] for ell_prime in axes
            if ell_prime != ell
        ]))
        base = Psis[ell] @ base

        for ell_prime in axes:
            if ell_prime == ell:
                continue
            # Sum along all axes but ell and ell_prime
            to_add = data.sum(axis=tuple([
                indices_dict[_ell] for _ell in axes
                if _ell != ell and _ell != ell_prime
            ]))
            
            # Multiply by Psi_{ell_prime} and then sum along ell_prime
            if indices_dict[ell_prime] < indices_dict[ell]:
                to_add = (lsum_Psis[ell_prime] @ to_add)
            else:
                to_add = (lsum_Psis[ell_prime] @ to_add.T)

            base += to_add

        return base

    b_bases = {
        ell: datatrans(ell, data, Psis)
        for ell in axes
    }
    max_cycles = 15
    for cycle in range(max_cycles):
        for ell in axes:
            # Preliminary calculations
            mean_lsum = (
                vec_kron_sum([
                    means[ell_prime]
                    for ell_prime in axes
                    if ell != ell_prime
                ])
                @ vec_kron_sum([
                    lsum_Psis[ell_prime]
                    for ell_prime in axes
                    if ell != ell_prime
                ])
            )

            b = (
                d_slashes[ell] * full_mean * lsum_Psis[ell]
                + full_mean * sum_Psis[ell]
                + mean_lsum
                - b_bases[ell]
            )
            A_inv_b = A_inv[ell] @ b
            means[ell] = (A_inv_b.sum() / A_inv[ell].sum()) * A_inv[ell].sum(axis=0) - A_inv_b
            
        full_mean = (
            (data.reshape(-1) - vec_kron_sum(list(means.values())))
            @ vec_kron_sum(list(lsum_Psis.values()))
            / sum(d_slashes[ell] * sum_Psis[ell] for ell in axes)
        )
    return means, full_mean