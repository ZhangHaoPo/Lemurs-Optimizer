# Lemur Optimization Algorithm
This project is a Python implementation of the Lemur Optimization Algorithm, refactored from an original [Matlab version](https://github.com/ammarabbasi/Lemurs-Optimizer). The objective is to find the best solution for different objective functions within given search ranges.

## Objective Functions

The objective functions are defined in the `ObjectiveFunctions` class. Each function has specific parameters: lower bound (`lb`), upper bound (`ub`), and dimension (`dim`), which represent the search range and number of variables, respectively. The objective function (`fobj`) is used to evaluate solutions.

### Available Functions

- `F1`: Sum of squares function
- `linear`: Linear function targeting `y = 36`
- `F18`: Complex multimodal function

## Usage

### Initialization

You need to specify the name of the objective function you want to optimize. The available functions are defined in the `ObjectiveFunctions` class.

### Example

Below is an example of how to use the Lemur Optimization Algorithm to optimize the `linear` function:

```python
if __name__ == '__main__':
    function_name = 'linear'
    optimizer = LemurOptimization(function_name)
    best_solution, final_results = optimizer.run_optimization()
    logger.info(f"best_solution = {best_solution}, final_results = {final_results}")