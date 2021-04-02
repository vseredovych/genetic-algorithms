import numpy as np
from typing import Callable


class MultiDimGA:
    def __init__(self):
        self.f = None
        self.h = None
        self.n = None
        self.dim = None
        self.intervals = None
        self.max_iter = None
        self.max_no_conv_iter = None
        self.tournament_n = None
        self.mutation_p = None
        self.crossbreeding_p = None
        self.max = None

        self.m = 0
        self.bit_array_sizes = []
        self.history = []

    def solve(self,
              f: Callable[[], float],
              intervals: list = [[-2, 2], [0, 1]],
              h: float = 1e-8,
              n: int = 100,
              tournament_n: int = 3,
              mutation_p: float = 0.1,
              crossbreeding_p: float = 0.9,
              max_iter: int = 100,
              max_no_conv_iter: int = 20,
              max: bool = False) -> float:

        self.h = h
        self.n = n
        self.f = f
        self.dim = np.array(intervals).shape[0]
        self.intervals = intervals
        self.tournament_n = tournament_n
        self.mutation_p = mutation_p
        self.crossbreeding_p = crossbreeding_p
        self.max_iter = max_iter
        self.max_no_conv_iter = max_no_conv_iter
        self.max = max

        self.bit_array_sizes = self._calc_bit_array_size()
        self.m = np.sum(self.bit_array_sizes)

        first_generation = np.random.choice(2, size=[self.n, self.m])

        args, args_eval = self._genetic_algorithm(first_generation)

        return args, f(*args)

    def _genetic_algorithm(self, generation: np.ndarray) -> tuple:
        self.history = []
        solution = np.inf
        no_conv_iter = 0

        for i in range(self.max_iter):
            generation = self._next_generation(generation)

            args = self._arg_eval_generation(generation)
            args_eval = self._eval_generation(generation)

            self.history.append(args_eval)

            if args_eval < solution:
                solution = args_eval
                no_conv_iter = 0
            else:
                no_conv_iter += 1

            if no_conv_iter > self.max_no_conv_iter:
                break

        return args, args_eval

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
        arguments = self._arg_eval_multi_dim(individual)
        y = self.f(*arguments)
        return -y if self.max else y

    def _arg_eval_single_dim(self, individual: np.ndarray, a: int, b: int, m: int) -> float:
        decimal = 0
        for idx, val in enumerate(individual):
            decimal += 2 ** idx * val

        return (b - a) / (2 ** m - 1) * decimal + a

    def _arg_eval_multi_dim(self, individual: np.ndarray) -> list:
        arguments = np.zeros(self.dim)

        for i in range(self.dim):
            a = self.intervals[i][0]
            b = self.intervals[i][1]
            m = self.bit_array_sizes[i]
            clip_from = sum(self.bit_array_sizes[:i])
            clip_to = sum(self.bit_array_sizes[:i+1])

            arguments[i] = self._arg_eval_single_dim(
                individual[clip_from:clip_to], a, b, m
            )

        return arguments

    def _eval_generation(self, generation: np.ndarray) -> float:
        return np.min([self._eval(ind) for ind in generation])

    def _arg_eval_generation(self, generation: np.ndarray) -> tuple:
        index = np.argmin([self._eval(ind) for ind in generation])
        return self._arg_eval_multi_dim(generation[index])

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

    def _calc_bit_array_single(self, interval: np.ndarray) -> int:
        a, b = interval[0], interval[1]
        samples = abs(b - a) / self.h
        m = 0
        while samples > 2 ** m:
            m += 1
        return m

    def _calc_bit_array_size(self) -> list:
        array_sizes = []

        for interval in self.intervals:
            array_sizes.append(self._calc_bit_array_single(interval))

        return array_sizes


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #f = lambda x, y: x * y
    f = lambda x, y: np.sin(10*x) + x * np.cos(2*np.pi*y)
    # f(x1,x2) = sin(10x1) +x1cos(2πx2), x1∈[−2,2],x2∈[0,1].

    single_dim_ga = MultiDimGA()
    res = single_dim_ga.solve(f)

    print(res)

    plt.plot(single_dim_ga.history)
    plt.show()


