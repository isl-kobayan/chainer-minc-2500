
from sklearn.manifold import TSNE as basicTSNE
import sklearn.manifold.t_sne as tsne
#from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np

class TSNE(basicTSNE):
    """
    unfix parameter ``n_iter_without_progress'' when using barnes-hut algorithm
    """
    def _tsne(self, P, degrees_of_freedom, n_samples, random_state,
              X_embedded=None, neighbors=None, skip_num_points=0):
        """Runs t-SNE."""
        # t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
        # and the Student's t-distributions Q. The optimization algorithm that
        # we use is batch gradient descent with three stages:
        # * early exaggeration with momentum 0.5
        # * early exaggeration with momentum 0.8
        # * final optimization with momentum 0.8
        # The embedding is initialized with iid samples from Gaussians with
        # standard deviation 1e-4.

        if X_embedded is None:
            # Initialize embedding randomly
            X_embedded = 1e-4 * random_state.randn(n_samples,
                                                   self.n_components)
        params = X_embedded.ravel()

        opt_args = {"n_iter": 50, "momentum": 0.5, "it": 0,
                    "learning_rate": self.learning_rate,
                    "n_iter_without_progress": self.n_iter_without_progress,
                    "verbose": self.verbose, "n_iter_check": 25,
                    "kwargs": dict(skip_num_points=skip_num_points)}
        if self.method == 'barnes_hut':
            m = "Must provide an array of neighbors to use Barnes-Hut"
            assert neighbors is not None, m
            obj_func = tsne._kl_divergence_bh
            objective_error = tsne._kl_divergence_error
            sP = squareform(P).astype(np.float32)
            neighbors = neighbors.astype(np.int64)
            args = [sP, neighbors, degrees_of_freedom, n_samples,
                    self.n_components]
            opt_args['args'] = args
            opt_args['min_grad_norm'] = 1e-3
            # uncomment only this line
            print('modified t-SNE')
            # opt_args['n_iter_without_progress'] = 30
            # Don't always calculate the cost since that calculation
            # can be nearly as expensive as the gradient
            opt_args['objective_error'] = objective_error
            opt_args['kwargs']['angle'] = self.angle
            opt_args['kwargs']['verbose'] = self.verbose
        else:
            obj_func = tsne._kl_divergence
            opt_args['args'] = [P, degrees_of_freedom, n_samples,
                                self.n_components]
            opt_args['min_error_diff'] = 0.0
            opt_args['min_grad_norm'] = self.min_grad_norm

        # Early exaggeration
        P *= self.early_exaggeration

        params, kl_divergence, it = tsne._gradient_descent(obj_func, params,
                                                      **opt_args)
        opt_args['n_iter'] = 100
        opt_args['momentum'] = 0.8
        opt_args['it'] = it + 1
        params, kl_divergence, it = tsne._gradient_descent(obj_func, params,
                                                      **opt_args)
        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations with early "
                  "exaggeration: %f" % (it + 1, kl_divergence))
        # Save the final number of iterations
        self.n_iter_final = it

        # Final optimization
        P /= self.early_exaggeration
        opt_args['n_iter'] = self.n_iter
        opt_args['it'] = it + 1
        params, error, it = tsne._gradient_descent(obj_func, params, **opt_args)

        if self.verbose:
            print("[t-SNE] Error after %d iterations: %f"
                  % (it + 1, kl_divergence))

        X_embedded = params.reshape(n_samples, self.n_components)
        self.kl_divergence_ = kl_divergence

        return X_embedded
