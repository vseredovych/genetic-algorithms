import numpy as np
from typing import Callable


class SingleDimGA:
    def __init__(self):
        self.f = None
        self.a = None
        self.b = None
        self.h = None
        self.n = None
        self.max_iter = None
        self.max_no_conv_iter = None
        self.tournament_n = None
        self.mutation_p = None
        self.crossbreeding_p = None

        self.m = None
        self.history = []

    def solve(self,
              f: Callable[[float], float],
              a: float = 0,
              b: float = 1,
              h: float = 1e-6,
              n: int = 100,
              tournament_n: int = 3,
              mutation_p: float = 0.1,
              crossbreeding_p: float = 0.9,
              max_iter: int = 100,
              max_no_conv_iter: int = 20) -> float:

        self.a = a
        self.b = b
        self.h = h
        self.n = n
        self.f = f
        self.tournament_n = tournament_n
        self.mutation_p = mutation_p
        self.crossbreeding_p = crossbreeding_p
        self.max_iter = max_iter
        self.max_no_conv_iter = max_no_conv_iter

        self.m = self._calc_bit_array_size()
        first_generation = np.random.choice(2, size=[self.n, self.m])

        return self._genetic_algorithm(first_generation)

    def _genetic_algorithm(self, generation: np.ndarray) -> tuple:
        self.history = []
        solution = np.inf
        no_conv_iter = 0

        for i in range(self.max_iter):
            generation = self._next_generation(generation)

            x = self._arg_eval_generation(generation)
            y = self._eval_generation(generation)

            self.history.append(y)

            if y < solution:
                solution = y
                no_conv_iter = 0
            else:
                no_conv_iter += 1

            if no_conv_iter > self.max_no_conv_iter:
                break

        return x, y

    def _next_generation(self, generation: np.ndarray) -> np.ndarray:
        next_generation = []

        while len(next_generation) < self.n:
            x1 = self._tournament(generation)
            x2 = self._tournament(generation)

            if np.random.rand() > self.mutation_p:
                x1, x2 = self._mutation(x1), self._mutation(x2)

            if np.random.rand() > self.crossbreeding_p:
                x1, x2 = self._crossbreeding(x1, x2)

            next_generation.extend([x1, x2])

        return np.array(next_generation).reshape(self.n, self.m)

    def _eval(self, individual: np.ndarray) -> float:
        x = self._arg_eval(individual)
        return self.f(x)

    def _arg_eval(self, individual: np.ndarray) -> float:
        decimal = 0
        for idx, val in enumerate(individual):
            decimal += 2 ** idx * val

        return (self.b - self.a) / (2 ** self.m - 1) * decimal + self.a

    def _eval_generation(self, generation: np.ndarray) -> float:
        return np.min([self._eval(ind) for ind in generation])

    def _arg_eval_generation(self, generation: np.ndarray) -> float:
        index = np.argmin([self._eval(ind) for ind in generation])
        return self._arg_eval(generation[index])

    def _tournament(self, generation: np.ndarray) -> np.ndarray:
        contestants_indices = np.random.randint(self.n, size=self.tournament_n)
        contestants = generation[contestants_indices]
        contestants_eval = [self._eval(contestant) for contestant in contestants]
        champion_index = np.argmin(contestants_eval)
        return contestants[champion_index]

    def _mutation(self, x: np.ndarray) -> np.ndarray:
        index = np.random.randint(self.m)
        x[index] = bool(x[index]) ^ 1
        return x

    def _crossbreeding(self, x1: np.ndarray, x2: np.ndarray) -> tuple:
        index = np.random.randint(self.m)
        buff = x1
        x1[index:] = x2[index:]
        x2[index:] = buff[index:]
        return x1, x2

    def _calc_bit_array_size(self) -> int:
        samples = abs(self.b - self.a) / self.h
        m = 0
        while samples > 2**m:
            m += 1
        return m


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    f = lambda x: x * np.sin(10 * np.pi * x) + 1

    single_dim_ga = SingleDimGA()
    res = single_dim_ga.solve(f)

    plt.plot(single_dim_ga.history)
    plt.show()
