import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.linalg import eigs
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx


def edge_overlap(A, B):
    """
    Compute edge overlap between two graphs (amount of shared edges).

    Args:
        A (sp.csr.csr_matrix): First input adjacency matrix.
        B (sp.csr.csr_matrix): Second input adjacency matrix.

    Returns:
        Edge overlap.
    """

    return A.multiply(B).sum() / 2


def link_prediction_performance(scores_matrix, val_ones, val_zeros):
    """
    Compute the link prediction performance of a score matrix on a set of validation edges and non-edges.

    Args:
        scores_matrix (np.array): Symmetric scores matrix of the graph generative model.
        val_ones (np.array): Validation edges. Rows represent indices of the input adjacency matrix with value 1. 
        val_zeros (np.array): Validation non-edges. Rows represent indices of the input adjacency matrix with value 0.
        
    Returns:
       2-Tuple containg ROC-AUC score and Average precision.
    """

    actual_labels_val = np.append(np.ones(len(val_ones)), np.zeros(len(val_zeros)))
    edge_scores = np.append(
        scores_matrix[val_ones[:, 0], val_ones[:, 1]],
        scores_matrix[val_zeros[:, 0], val_zeros[:, 1]],
    )
    return (
        roc_auc_score(actual_labels_val, edge_scores),
        average_precision_score(actual_labels_val, edge_scores),
    )


def scores_matrix_from_transition_matrix(transition_matrix, symmetric=True):
    """
    Compute the scores matrix from the transition matrix.

    Args:
        transition_matrix (np.array, shape=(N,N)): Matrix whose entries (i,j) correspond to the probability of a 
                                                   transition from node i to j.
        symmetric (bool, default:True): If True, symmetrize the resulting scores matrix.

    Returns:
        scores_matrix(sp.csr.csr_matrix, shape=(N, N)): Matrix whose entries (i,j) correspond to the weight of the 
                                                        directed edge (i, j) in an edge-independent model.
    """

    N = transition_matrix.shape[0]
    p_stationary = np.real(eigs(transition_matrix.T, k=1, sigma=0.99999)[1])
    p_stationary /= p_stationary.sum()
    scores_matrix = np.maximum(p_stationary * transition_matrix, 0)

    if symmetric:
        scores_matrix += scores_matrix.T

    return scores_matrix


def graph_from_scores(scores_matrix, n_edges):
    """
    Assemble a symmetric binary graph from the input score matrix. Ensures that there will be no singleton nodes.
    See the paper for details.

    Args:
        scores_matrix (sp.csr.csr_matrix, shape=(N, N)): Matrix whose entries (i,j) correspond to the weight of the 
                                                        directed edge (i, j) in an edge-independent model.
        n_edges (int): The desired number of edges in the generated graph.

    Returns
    -------
    target_g (sp.csr.csr_matrix, shape=(N, N)): Adjacency matrix of the generated graph.
    """

    target_g = sp.csr_matrix(scores_matrix.shape)

    np.fill_diagonal(scores_matrix, 0)

    degrees = scores_matrix.sum(1)  # The row sum over the scores_matrix.

    N = scores_matrix.shape[0]

    for n in range(N):  # Iterate over the nodes
        target = np.random.choice(N, p=scores_matrix[n] / degrees[n])
        target_g[n, target] = 1
        target_g[target, n] = 1

    diff = np.round((2 * n_edges - target_g.sum()) / 2)
    if diff > 0:
        triu = np.triu(scores_matrix)
        triu[target_g.nonzero()] = 0
        triu = triu / triu.sum()

        triu_ixs = np.triu_indices_from(scores_matrix)
        extra_edges = np.random.choice(
            triu_ixs[0].shape[0], replace=False, p=triu[triu_ixs], size=int(diff)
        )

        target_g[(triu_ixs[0][extra_edges], triu_ixs[1][extra_edges])] = 1
        target_g[(triu_ixs[1][extra_edges], triu_ixs[0][extra_edges])] = 1

    return target_g
