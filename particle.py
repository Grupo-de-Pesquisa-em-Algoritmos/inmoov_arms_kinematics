import numpy as np
import random as rand

class Particle:
    def _init_(self, pos, vel, in_fac, cog_fac, soc_fac, fitness, q_min, q_max, r):
        self.pos = pos
        self.vel = vel
        self.in_fac = in_fac
        self.cog_fac = cog_fac
        self.soc_fac = soc_fac
        self.fitness = fitness
        self.q_min = q_min
        self.q_max = q_max
        self.r = r

        self.p_best = np.full(len(r.links), np.inf)


    def updateFitness(self):
        forward_kin = self.r.fkine(self.pos)
        end_ef_position = forward_kin[:, -1]
        self.fitness = np.linalg.norm(super().goal - end_ef_position)

    def updateVelocity(self):
        r1 = rand.random()
        r2 = rand.random()
        self.vel = self.in_fac * self.vel + self.cog_fac * r1 * (self.p_best[1] - self.pos) + self.soc_fac * r2 * (super().g_best[1] - self.pos)

    def updatePosition(self):
        self.pos = self.pos + self.vel
        if any(self.position > self.q_max) or any(self.position < self.q_min):
                self.position = max(self.q_min, min(self.position, self.q_max));