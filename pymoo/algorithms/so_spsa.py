from pymoo.model.algorithm import Algorithm
from pymoo.model.mutation import Mutation
from pymoo.util.display import SingleObjectiveDisplay

import numpy as np


class SPSA(Algorithm):
    """
    Simultaneous Perturbation Stochastic Approximation
    """

    def __init__(self, pert_size, step_size, sampling, display=SingleObjectiveDisplay(), **kwargs) -> None:
        super().__init__(display=display, **kwargs)
        self.pert_size = pert_size
        self.step_size = step_size
        self.sampling = sampling

    def setup(self, problem, termination=None, callback=None, display=None, seed=None, verbose=False,
              save_history=False, pf=True, evaluator=None, **kwargs):
        super().setup(problem, termination, callback, display, seed, verbose, save_history, True, pf,
                      evaluator, **kwargs)

    def _initialize(self):
        self.pop = self.sampling.do(self.problem, 1)

    def _next(self):
        self.pop = SPSAMutation(self.pert_size, self.step_size).do(self.problem, self.pop)
        out = self.problem.evaluate(self.pop.get("X"), return_values_of=["F", "feasible"], return_as_dictionary=True)
        for k, v in out.items():
            self.pop.set(k, v)
        self.evaluator.n_eval += 3 * len(self.pop)

    def _set_optimum(self, force=False):
        self.opt = self.pop


class SPSAMutation(Mutation):
    """
    Implements SPSA as a Mutation object for other algorithms to use
    """

    def __init__(self, pert_size=0.01, step_size=0.1) -> None:
        super().__init__()
        self.pert_size = pert_size
        self.step_size = step_size

    def _do(self, problem, X, **kwargs):
        # generate perturbation vector using a suitable distribution (here Rademacher)
        c_delta = self.pert_size * np.random.choice([-1, 1], X.shape)

        # perturb the parameter vectors
        X1 = X + c_delta
        X2 = X - c_delta

        # compute scores
        J1 = problem.evaluate(X1)
        J2 = problem.evaluate(X2)

        # compute the approximate gradient
        G = (J1 - J2) / (2*c_delta)

        # simple gradient descent step
        return X - self.step_size * G
