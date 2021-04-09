from typing import Callable
from ipywidgets import IntProgress

import numpy as np
import copy

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
        self.crossover_p = None
        self.min_lifetime = None
        self.max_lifetime = None
        self.reproduction_p = None
        self.max_problem = None
        self.verbose = None
        
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
              crossover_p: float = 0.9,
              max_iter: int = 100,
              max_no_conv_iter: int = 20,
              min_lifetime: int = 10,
              max_lifetime: int = 10,
              reproduction_p: float = 0.5,
              max_problem: bool = False,
              verbose: bool = False) -> float:

        self.h = h
        self.n = n
        self.f = f
        self.dim = np.array(intervals).shape[0]
        self.intervals = intervals
        self.tournament_n = tournament_n
        self.mutation_p = mutation_p
        self.crossover_p = crossover_p
        self.max_iter = max_iter
        self.max_no_conv_iter = max_no_conv_iter
        self.max_problem= max_problem
        self.verbose = verbose
        self.progress_bar = IntProgress(max=max_iter)
        
        self.min_lifetime = min_lifetime
        self.max_lifetime = max_lifetime
        self.eta = 1/2 * (self.max_lifetime - self.min_lifetime)
        self.reproduction_p = reproduction_p
        
        
        self.bit_array_sizes = self._calc_bit_array_size()
        self.m = np.sum(self.bit_array_sizes)

        args, solution = self._genetic_algorithm()
        return args, f(*args)

    def _genetic_algorithm(self) -> tuple:
        self.history = []
        best_score = np.inf
        best_args = None
        no_conv_iter = 0
        
        display(self.progress_bar)
        
        population = self._initialize_population()
        for i in range(self.max_iter):
            population = self._next_population(population)

            if population.shape[0] == 0:
                break

            args, score = self._eval_population(population)            

            self.history.append(score)

            if (i+1) % 1 == 0:
                self.progress_bar.value = i + 1
            
            if score < best_score:
                best_score = score
                best_args = args
                no_conv_iter = 0
            else:
                no_conv_iter += 1

            if no_conv_iter > self.max_no_conv_iter:
                break

        self.progress_bar.value = self.max_iter
        return best_args, best_score
    
    def _initialize_population(self):
        population_values = np.random.choice(2, size=[self.n, self.m])
        population = np.array([{'age': 0, 'value': ind} for ind in population_values])
        return population

    def _next_population(self, population: np.ndarray) -> np.ndarray:
        population_lifetime = self._calc_lifetime(population)
        survived = []

        for ind in population:
            ind['age'] = ind['age'] + 1

        for ind, lifetime in zip(population, population_lifetime):
            if ind['age'] < lifetime:
                survived.append(ind)
                
        died = len(population) - len(survived)

        offspring = self._spawn_offspring(np.array(survived))
        new_population = np.concatenate([survived + offspring])
        
        if self.verbose:
            print(f"Total: {len(new_population)}; Born: {len(offspring)}; Died: {died}")
        
        return new_population

    def _spawn_offspring(self, population):
        offspring = []
        
        for _ in range(int(population.shape[0]/2)):
            if np.random.rand() < self.reproduction_p:
                x1 = self._tournament(population)
                x2 = self._tournament(population)

                if np.random.rand() < self.mutation_p:
                    x1, x2 = self._mutation(x1), self._mutation(x2)

                if np.random.rand() < self.crossover_p:
                    x1, x2 = self._crossover(x1, x2)
                
                x1['age'], x2['age'] = 0, 0
                offspring.extend([x1, x2])

        return offspring

    def _calc_lifetime(self, population):
        scores = - np.array([self._eval(ind) for ind in population])
        scores_ = scores - np.min(scores) + 1
        
        scores_min_abs = np.min(np.abs(scores))
        scores_max_abs = np.max(np.abs(scores_))

        lifetime = self.min_lifetime + 2*self.eta*(scores_ - scores_max_abs)/(scores_max_abs - scores_min_abs)
        return lifetime
    
    def _binary_to_decimal(self, bits: np.ndarray, a: int, b: int, m: int) -> float:
        decimal = 0
        for idx, bit in enumerate(bits):
            decimal += 2 ** idx * bit

        return (b - a) / (2 ** m - 1) * decimal + a
    
    def _eval(self, individual: dict) -> float:
        args = self._arg_eval(individual)
        y = self.f(*args)
        return -y if self.max_problem else y

    def _arg_eval(self, individual: dict) -> list:
        args = np.zeros(self.dim)

        for i in range(self.dim):
            a = self.intervals[i][0]
            b = self.intervals[i][1]
            m = self.bit_array_sizes[i]
            clip_from = sum(self.bit_array_sizes[:i])
            clip_to = sum(self.bit_array_sizes[:i+1])

            args[i] = self._binary_to_decimal(
                individual['value'][clip_from:clip_to], a, b, m
            )

        return args

    def _eval_population(self, population: np.ndarray) -> tuple:
        index = np.argmin([self._eval(ind) for ind in population])

        ind_args = self._arg_eval(population[index])
        ind_score = self._eval(population[index])

        return ind_args, ind_score

    def _tournament(self, population: np.ndarray) -> np.ndarray:
        contestants_indices = np.random.randint(population.shape[0], size=self.tournament_n)
        contestants = population[contestants_indices]
        contestants_eval = [self._eval(contestant) for contestant in contestants]
        champion_index = np.argmin(contestants_eval)
        return copy.deepcopy(contestants[champion_index])

    def _mutation(self, x: dict) -> np.ndarray:
        index = np.random.randint(self.m)
        x['value'][index] = bool(x['value'][index]) ^ True
        return x

    def _crossover(self, x1: dict, x2: dict) -> tuple:
        points = np.random.randint(self.m, size=(2))
        x1['value'][min(points):max(points)], x2['value'][min(points):max(points)] = \
        x2['value'][min(points):max(points)], x1['value'][min(points):max(points)]
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

