import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx



def train_val_test_split(A, val_share, test_share, seed=123):
    """
    Split the edges of the input graph in training-, validation-, and test set.
    
    Randomly split a share of the edges of the input graph for validation- and test set, while ensuring that the 
    remaining graph stays connected. Additionally choose an equal amount of non-edges from the input graph.

    Args:
        A (sp.csr.csr_matrix): The input adjacency matrix.
        val_share: Fraction of edges that form the validation set.
        test_share: Fraction of edges that form the test set.
        seed: Random seed.
        
    Returns:
        train_graph (sp.csr.csr_matrix): Remaining graph after split, which is used for training.
        val_ones (np.array): Validation edges. Rows represent indices of the input adjacency matrix with value 1. 
        val_zeros (np.array): Validation non-edges. Rows represent indices of the input adjacency matrix with value 0.
        test_ones (np.array): Test edges. Rows represent indices of the input adjacency matrix with value 1. 
        test_zeros (np.array): Test non-edges. Rows represent indices of the input adjacency matrix with value 0.
    """  
    
    np.random.seed(seed)
    G = nx.from_scipy_sparse_matrix(A)
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Test symmetric, connected and has sufficiently many edges
    assert (abs(A-A.T)>1e-10).nnz == 0, 'Graph is not symmetric.'
    assert nx.is_connected(G), 'Graph is not connected.'
    assert num_edges - num_nodes > (val_share + test_share) * num_edges + 1, 'Val- and test-share are too large.'    
    
    # Ensure that train graph is symmetric by protecting certain edges
    # Split edges into val-, test- and training-set
    protected_edges = list(nx.minimum_spanning_tree(G).edges())
    edges_set = set(G.edges())
    free_edges = list(edges_set - set(protected_edges))
    np.random.shuffle(free_edges)
    num_val = int(val_share * num_edges)
    num_test = int(test_share * num_edges)
    val_ones = np.array(free_edges[:num_val])
    test_ones = np.array(free_edges[num_val:num_val+num_test])
    train_edges = free_edges[num_val+num_test:] + protected_edges

    G_train = nx.Graph()
    G_train.add_nodes_from(G)
    G_train.add_edges_from(train_edges)
    train_graph = nx.to_scipy_sparse_matrix(G_train)
    
    # Draw non-edges from input graph: draw random tuples, remove direction, loops, and input edges
    non_edges = np.random.choice(num_nodes, size=(2*(num_val+num_test), 2))
    non_edges = np.sort(non_edges[non_edges[:, 0] != non_edges[:, 1]]) # Remove loops and direction
    non_edges = np.unique(non_edges, axis=0) # Remove multiple edges
    non_edges = [tuple(edge) for edge in non_edges if tuple(edge) not in edges_set] # Remove input edges
    np.random.shuffle(non_edges)
    assert len(non_edges)>= num_val + num_test, 'Too few non-zero edges.'
    val_zeros = np.array(non_edges[:num_val])
    test_zeros = np.array(non_edges[num_val:num_val+num_test])
    return train_graph, val_ones, val_zeros, test_ones, test_zeros


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
    edge_scores = np.append(scores_matrix[val_ones[:,0], val_ones[:,1]],
                            scores_matrix[val_zeros[:,0], val_zeros[:,1]])
    return roc_auc_score(actual_labels_val, edge_scores), average_precision_score(actual_labels_val, edge_scores)