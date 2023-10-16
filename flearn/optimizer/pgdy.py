from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf


class PerturbedGradientDescent(optimizer.Optimizer):
    """Implementation of Perturbed Gradient Descent, i.e., FedProx optimizer"""

    def __init__(self, learning_rate: object = 0.001, mu: object = 0.01, use_locking: object = False,
                 name: object = "PGD") -> object:
        super(PerturbedGradientDescent, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._mu = mu

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._mu_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._mu_t = ops.convert_to_tensor(self._mu, name="prox_mu")
        #self._zeros_slot(tf.Variable(1.), "tier", self._name)
    def _create_slots(self, var_list):
        # Create slots for the global solution.
        for v in  var_list:
            self._zeros_slot(v, "vstar", self._name)
        #self._zeros_slot( tf.Variable(1.), "tier",self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")
        #tier = self.get_slot(var, "tier")

        var_update = state_ops.assign_sub(var, lr_t * (grad + mu_t * (var - vstar)))

        return control_flow_ops.group(*[var_update, ])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")
        #tier = self.get_slot(var, "tier")

        v_diff = state_ops.assign(vstar, mu_t * (var - vstar), use_locking=self._use_locking)

        with ops.control_dependencies([v_diff]):  # run v_diff operation before scatter_add
            scaled_grad = scatter_add(vstar, indices, grad)
        var_update = state_ops.assign_sub(var, lr_t  * scaled_grad)

        return control_flow_ops.group(*[var_update, ])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(x, i, v))

    def set_params(self, cog, client,tier):
        with client.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, cog):
                vstar = self.get_slot(variable, "vstar")
                #tier = self.get_slot(variable, "tier")
                vstar.load(value, client.sess)
                #self._lr=tier
                #self._lr.assign(tier)
