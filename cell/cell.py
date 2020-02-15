import abc
import time
import numpy as np
import torch
DEVICE = 'cpu'
DTYPE = torch.float32


class Callback(abc.ABC):
    def __init__(self, invoke_every):
        self.training_stopped = False
        self.invoke_every = invoke_every
        
    def __call__(self, loss, model):
        if model.step % self.invoke_every == 0:
            self.invoke(loss, model)
        
    def stop_training(self):
        self.training_stopped = True
    
    @abc.abstractmethod
    def invoke(self, loss, model):
        pass


class LinkPredictionCriterion(Callback):
    def __init__(self, invoke_every, val_ones, val_zeros, max_patience):
        super().__init__(invoke_every)
        self.val_ones = val_ones
        self.val_zeros = val_zeros
        self.max_patience = max_patience

        self.patience = 0
        self.best_model = None
        self.best_link_pred_score = 0.

    def invoke(self, loss, model):
        """"""
        start = time.time()
        model.update_scores_matrix()
        roc_auc, avg_prec = utils.link_prediction_performance(model._scores_matrix,
                                                              self.val_ones,
                                                              self.val_zeros)

        link_pred_time = time.time() - start
        model.total_time += link_pred_time
        
        step_str = f'{model.step:{model.step_str_len}d}'
        print(f'Step: {step_str}/{model.steps}',
              f'Loss: {loss:.5f}',
              f'ROC-AUC Score: {roc_auc:.3f}',
              f'Average Precision: {avg_prec:.3f}'
              f'Total-Time: {int(model.total_time)}')
        link_pred_score = roc_auc + avg_prec
        
        if link_pred_score > self.best_link_pred_score:
            self.best_link_pred_score = link_pred_score
            self.best_scores_matrix = model._scores_matrix.copy()
            self.patience = 0
            
        elif self.patience >= self.max_patience:
            self.stop_training()
            
        else:
            self.patience += 1
            
        model._scores_matrix = self.best_scores_matrix

class EdgeOverlapCriterion(Callback):
    """
    This callback serves in three ways:
    - It tracks the EdgeOverlap and stops if the limit is met.
    - It tracks the validation AUC-ROC score and the average precision.
    - It tracks the total time.
    """
    def __init__(self, invoke_every, EO_limit=1.):
        super().__init__(invoke_every)
        self.EO_limit = EO_limit

    def invoke(self, loss, model):
        start = time.time()
        model.update_scores_matrix()
        sampled_graph = model.sample_graph()
        overlap = utils.edge_overlap(model.A_sparse, sampled_graph) / model.num_edges
        overlap_time = time.time() - start
        model.total_time += overlap_time
        
        step_str = f'{model.step:{model.step_str_len}d}'
        print(f'Step: {step_str}/{model.steps}',
              f'Loss: {loss:.5f}',
              f'Edge-Overlap: {overlap:.3f}',
              f'Total-Time: {int(model.total_time)}')
        if overlap >= self.EO_limit:
            self.stop_training()


class Cell(object):
    def __init__(self, A, H, loss_fn=None, callbacks=[]):
        self.num_edges = A.sum()/2
        self.A_sparse = A
        self.A = torch.tensor(A.toarray())
        self.step = 1
        self.callbacks = callbacks
        self._optimizer = None
        
        N = A.shape[0]
        gamma = np.sqrt(2/(N+H))
        self.W_down = (gamma * torch.randn(N, H, device=DEVICE, dtype=DTYPE)).clone().detach().requires_grad_()
        self.W_up = (gamma * torch.randn(H, N, device=DEVICE, dtype=DTYPE)).clone().detach().requires_grad_()
        
        if loss_fn:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = self.built_in_loss_fn
        
        self.total_time = 0
        self.scores_matrix_needs_update = True
              
    def __call__(self):
        return torch.nn.functional.softmax(self.get_W(), dim=-1).detach().numpy()
    
    def get_W(self):
        W = torch.mm(self.W_down, self.W_up)
        W -= W.max(dim=-1, keepdims=True)[0]
        return W
    
    def built_in_loss_fn(self, W, A, num_edges):
        """
        Computes the weighted cross-entropy loss in logits with weight matrix M * P.
        Parameters
        ----------
        W: torch.tensor of shape (N, N)
                Logits of learnable (low rank) transition matrix.

        Returns
        -------
        loss: torch.tensor (float)
                Loss at logits.
        """
        d = torch.log(torch.exp(W).sum(dim=-1, keepdims=True))
        loss = .5 * torch.sum(A * (d * torch.ones_like(A) - W)) / num_edges
        return loss
    
    def _closure(self):
        W = self.get_W()
        loss = self.loss_fn(W=W, A=self.A, num_edges=self.num_edges)
        self._optimizer.zero_grad()
        loss.backward()
        return loss
        
    def _train_step(self):
        time_start = time.time()
        loss = self._optimizer.step(self._closure)
        time_end = time.time()
        return loss.item(), (time_end - time_start)
    
    def train(self, steps, optimizer_fn, optimizer_args, EO_criterion=None):
        self._optimizer = optimizer_fn([self.W_down, self.W_up], **optimizer_args)
        self.steps = steps
        self.step_str_len = len(str(steps))
        self.scores_matrix_needs_update = True
        stop = False
        for self.step in range(self.step, steps+self.step):
            loss, time = self._train_step()
            self.total_time += time
            for callback in self.callbacks:
                callback(loss=loss, model=self)
                stop = stop or callback.training_stopped    
            if stop: break
    
    def update_scores_matrix(self):
        self._scores_matrix = utils.scores_matrix_from_transition_matrix(transition_matrix=self(),
                                                                         symmetric=True)
        self.scores_matrix_needs_update = False

    def sample_graph(self):
        if self.scores_matrix_needs_update:
            self.update_scores_matrix()
            
        sampled_graph = utils.graph_from_scores(self._scores_matrix, self.num_edges)
        return sampled_graph
