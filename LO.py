import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectiveFunctions:
    @staticmethod
    def F1(x):
        return np.sum(x ** 2)
    
    @staticmethod
    def linear(x):
        target_y = 36
        y = 2 * x + 5
        return abs(y - target_y)
    
    @staticmethod
    def F18(x):
        return (1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * (x[0] ** 2) - 14 * x[1] + 6 * x[0] * x[1] + 3 * (x[1] ** 2))) * \
            (30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 * (x[0] ** 2) + 48 * x[1] - 36 * x[0] * x[1] + 27 * (x[1] ** 2)))

    functions_info = {
        'F1': {'fobj': F1.__func__, 'lb': -100, 'ub': 100, 'dim': 30},
        'linear': {'fobj': linear.__func__, 'lb': -100, 'ub': 100, 'dim': 1},
        'F18': {'fobj': F18.__func__, 'lb': -2, 'ub': 0, 'dim': 2}
    }

    @staticmethod
    def get_function_details(function_name):
        if function_name not in ObjectiveFunctions.functions_info:
            raise ValueError('Invalid function name')
        func_info = ObjectiveFunctions.functions_info[function_name]
        return func_info['lb'], func_info['ub'], func_info['dim'], func_info['fobj']

class LemurOptimization:
    def __init__(self, function_name, PopSize=30, jumping_rate_min=0.1, jumping_rate_max=0.5, Max_iter=10000):
        self._msg = "[LemurOptimization]"
        self.function_name = function_name
        self.PopSize = PopSize
        self.jumping_rate_min = jumping_rate_min
        self.jumping_rate_max = jumping_rate_max
        self.Max_iter = Max_iter
        self.ObjVal = np.zeros(self.PopSize)
        self.BestResults = np.zeros((1, 1))
        self.lb, self.ub, self.dim, self.fobj = ObjectiveFunctions.get_function_details(self.function_name)

    def initialization(self):
        Boundary_no = np.size(self.ub)
        Positions = np.zeros((self.PopSize, self.dim))
        if Boundary_no == 1:
            Positions = np.random.rand(self.PopSize, self.dim) * (self.ub - self.lb) + self.lb
        elif Boundary_no > 1:
            for i in range(self.dim):
                ub_i = self.ub[i]
                lb_i = self.lb[i]
                Positions[:, i] = np.random.rand(self.PopSize) * (ub_i - lb_i) + lb_i
        return Positions

    def calculateFitness(self, fObjV):
        if np.isscalar(fObjV):
            fObjV = np.array([fObjV])
        fFitness = np.zeros_like(fObjV)
        ind = np.where(fObjV >= 0)
        fFitness[ind] = 1.0 / (fObjV[ind] + 1)
        ind = np.where(fObjV < 0)
        fFitness[ind] = 1 + np.abs(fObjV[ind])
        return fFitness

    def run_optimization(self):
        swarm = self.initialization()
        self.ObjVal = np.zeros(self.PopSize)

        for i in range(self.PopSize):
            self.ObjVal[i] = self.fobj(swarm[i][:])

        Fitness = self.calculateFitness(self.ObjVal)

        itr = 0
        conv = np.zeros((1, self.Max_iter))

        while itr < self.Max_iter:
            jumping_rate = self.jumping_rate_max - itr * ((self.jumping_rate_max - self.jumping_rate_min) / self.Max_iter)
            sorted_indexes = np.argsort(Fitness)

            for i in range(self.PopSize):
                current_solution = np.where(sorted_indexes == i)[0][0]
                near_solution_postion = current_solution - 1 if current_solution != 0 else 0
                near_solution = sorted_indexes[near_solution_postion]

                cost = np.min(self.ObjVal)
                best_solution_Index = np.argmin(self.ObjVal)

                NewSol = np.array(swarm[i, :])

                for j in range(self.dim):
                    r = np.random.rand()
                    if r < jumping_rate:
                        NewSol[j] = swarm[i, j] + np.abs(swarm[i, j] - swarm[near_solution][j]) * (np.random.rand() - 0.5) * 2
                        NewSol[j] = np.minimum(np.maximum(NewSol[j], self.lb if np.size(self.lb) == 1 else self.lb[j]), self.ub if np.size(self.ub) == 1 else self.ub[j])
                    else:
                        NewSol[j] = swarm[i, j] + np.abs(swarm[i, j] - swarm[best_solution_Index][j]) * (np.random.rand() - 0.5) * 2
                        NewSol[j] = np.minimum(np.maximum(NewSol[j], self.lb if np.size(self.lb) == 1 else self.lb[j]), self.ub if np.size(self.ub) == 1 else self.ub[j])

                ObjValSol = self.fobj(NewSol)
                FitnessSol = self.calculateFitness(ObjValSol)

                if self.ObjVal[i] > ObjValSol:
                    swarm[i][:] = NewSol
                    Fitness[i] = FitnessSol
                    self.ObjVal[i] = ObjValSol

            if itr % 100 == 0:
                min_index = np.argmin(self.ObjVal)
                best_solution = swarm[min_index, :]
                logger.info(f"{self._msg} Itr {itr}, Results {np.min(self.ObjVal)}")
                logger.info(f"{self._msg} Best Solution:")
                for i in range(self.dim):
                    logger.info(f"{self._msg} x[{i+1}] = {best_solution[i]}")
                logger.info(f'=======================')

            conv[:, itr] = np.min(self.ObjVal)
            itr += 1

        min_index = np.argmin(self.ObjVal)
        best_solution = swarm[min_index, :]
        self.BestResults = np.min(self.ObjVal)

        return best_solution, self.BestResults, 


if __name__ == '__main__':
    function_name = 'linear'
    optimizer = LemurOptimization(function_name)
    best_solution, final_results = optimizer.run_optimization()
    logger.info(f"best_solution = {best_solution}, final_results = {final_results}")
