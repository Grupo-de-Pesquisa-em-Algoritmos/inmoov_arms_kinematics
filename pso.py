import numpy as np
import particle

class PSO:
    def __init__(self, n_part, n_iter, in_fac, cog_fac, soc_fac, parts, r, q_min, q_max, tol, goal):
        self.n_part = n_part
        self.n_iter = n_iter
        self.in_fac = in_fac
        self.cog_fac = cog_fac
        self.soc_fac = soc_fac
        self.parts = parts
        self.r = r
        self.q_min = q_min
        self.q_max = q_max
        self.tol = tol
        self.goal = goal

        self.g_best = np.full(len(r.links), np.inf)

    def initParticles(self):
        for i in range(self.n_part):
            pass
