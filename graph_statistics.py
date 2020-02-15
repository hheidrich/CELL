import numpy as np
import scipy.sparse as sp
import networkx as nx
import powerlaw



def max_degree(A):
    """Compute max degree."""
    degrees = A.sum(axis=-1)
    return np.max(degrees)


def min_degree(A):
    """Compute min degree."""
    degrees = A.sum(axis=-1)
    return np.min(degrees)


def average_degree(A):
    """Compute average degree."""
    degrees = A.sum(axis=-1)
    return np.mean(degrees)


def LCC(A):
    """
    Compute the size of the largest connected component (LCC)

    Parameters
    ----------
    A: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Size of LCC

    """
    G = nx.from_scipy_sparse_matrix(A)
    return max([len(c) for c in nx.connected_components(G)])


def wedge_count(A):
    """
    Compute the wedge count of the input graph

    Parameters
    ----------
    A: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    The wedge count.
    """

    degrees = np.array(A.sum(axis=-1))
    return 0.5 * np.dot(degrees.T, degrees-1).reshape([])


def claw_count(A):
    """
    Compute the claw count of the input graph

    Parameters
    ----------
    A: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Claw count
    """

    degrees = np.array(A.sum(axis=-1))
    return 1/6 * np.sum(degrees * (degrees-1) * (degrees-2))


def triangle_count(A):
    """
    Compute the triangle count of the input graph

    Parameters
    ----------
    A: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Triangle count
    """

    A_graph = nx.from_scipy_sparse_matrix(A)
    triangles = nx.triangles(A_graph)
    t = np.sum(list(triangles.values())) / 3
    return int(t)


def square_count(A):
    """
    Compute the square count of the input graph

    Parameters
    ----------
    A: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Square count
    """

    A_squared = A @ A
    common_neighbors = sp.triu(A_squared, k=1).tocsr()
    num_common_neighbors = np.array(common_neighbors[common_neighbors.nonzero()]).reshape(-1)
    return np.dot(num_common_neighbors, num_common_neighbors-1) / 4


def power_law_alpha(A):
    """
    Compute the power law coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Power law coefficient
    """

    degrees = np.array(A.sum(axis=-1)).flatten()
    return powerlaw.Fit(degrees, xmin=max(np.min(degrees),1), verbose=False).power_law.alpha


def gini(A):
    """
    Compute the Gini coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Gini coefficient
    """
    N = A.shape[0]
    degrees_sorted = np.sort(np.array(A.sum(axis=-1)).flatten())
    return 2 * np.dot(degrees_sorted, np.arange(1, N+1)) / (N * np.sum(degrees_sorted)) - (N+1) / N


def edge_distribution_entropy(A):
    """
    Compute the relative edge distribution entropy of the input graph.

    Parameters
    ----------
    A: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Rel. edge distribution entropy
    """
    N = A.shape[0]
    degrees = np.array(A.sum(axis=-1)).flatten()
    degrees /= degrees.sum()
    return -np.dot(np.log(degrees), degrees) / np.log(N)


def assortativity(A):
    """Compute assortativity."""
    G = nx.from_scipy_sparse_matrix(A)
    return nx.degree_assortativity_coefficient(G)


def clustering_coefficient(A):
    """Compute clustering coefficient."""
    return 3 * triangle_count(A) / claw_count(A)


def cpl(A):
    """Compute characteristic path length."""
    P = sp.csgraph.shortest_path(A)
    return P[((1 - np.isinf(P)) * (1 - np.eye(P.shape[0]))).astype(np.bool)].mean()


def compute_graph_statistics(A):
    """

    Parameters
    ----------
    A: sparse matrix
          The input adjacency matrix.
          
    Returns
    -------
    Dictionary containing the following statistics:
             * Maximum, minimum, mean degree of nodes
             * Size of the largest connected component (LCC)
             * Wedge count
             * Claw count
             * Triangle count
             * Square count
             * Power law exponent
             * Gini coefficient
             * Relative edge distribution entropy
             * Assortativity
             * Clustering coefficient
             * Characteristic path length
    """
    statistics = {}

    # Degree statistics
    statistics['d_max'] = max_degree(A)
    statistics['d_min'] = min_degree(A)
    statistics['d'] = average_degree(A)
    # largest connected component
    statistics['LCC'] = LCC(A)
    # wedge count
    statistics['wedge_count'] = wedge_count(A)
    # claw count
    statistics['claw_count'] = claw_count(A)
    # triangle count
    statistics['triangle_count'] = triangle_count(A)
    # Square count
    statistics['square_count'] = square_count(A)
    # power law exponent
    statistics['power_law_exp'] = power_law_alpha(A)
    # gini coefficient
    statistics['gini'] = gini(A)
    # Relative edge distribution entropy
    statistics['rel_edge_distr_entropy'] = edge_distribution_entropy(A)
    # Assortativity
    statistics['assortativity'] = assortativity(A)
    # Clustering coefficient
    statistics['clustering_coefficient'] = clustering_coefficient(A)
    # Characteristic path length
    statistics['cpl'] = cpl(A)
    return statistics