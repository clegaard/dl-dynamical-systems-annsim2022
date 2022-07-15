from typing import Literal
import torch
import torch.nn as nn


# class SoftEqualityConstraint:
#     def __init__(self, lhs, rhs):
#         self.weight = 1e3
#         self.lhs = lhs
#         self.rhs = rhs

#     def __call__(self, *args, **kwargs):
#         pass


class EqualityConstraint:
    def __init__(self, lhs, rhs) -> None:
        self._lhs = lhs
        self._rhs = rhs

    def __call__(self, *args, **kwargs) -> bool:
        # positional and keyword args can't be dispatched to the lhs and rhs in a meaningful way based on overloading the call operator
        # it would be a requirement that the lhs and rhs capture all their arguments in the function
        print(
            f"lhs is {self._lhs(*args, **kwargs)} and rhs is {self._rhs(*args, **kwargs)}"
        )
        return self._lhs(*args, **kwargs) == self._rhs(*args, **kwargs)


class Expression:
    def __init__(self, callable):
        self._callable = callable

    def __call__(self, *args, **kwargs):
        return self._callable(*args, **kwargs)

    def __eq__(self, __o: object) -> EqualityConstraint:
        return EqualityConstraint(self, __o)


def expressionfy(callable):
    return Expression(callable)


if __name__ == "__main__":

    network = nn.Sequential(
        nn.Linear(2, 10),
    )

    def parameter_lhs():
        return torch.linalg.eigvals(network[0].weight.T @ network[0].weight).sum()

    def parameter_rhs():
        return 0.0

    # if we need sympy/cvxpy like api, we need to overload operators of the callable. There are two ways to do this
    # 1) we wrap the callable inside another class using function, similar API to sympy's lambdify or JAX jit or grad
    parameter_lhs = expressionfy(parameter_lhs)
    parameter_lhs = expressionfy(parameter_lhs)

    parameter_constraint = parameter_lhs == parameter_rhs
    parameter_constraint()

    # 1b) we wrap using decorator syntax
    @expressionfy
    def parameter_lhs():
        return torch.linalg.eigvals(network[0].weight.T @ network[0].weight).sum()

    @expressionfy
    def parameter_rhs():
        return 0.0

    parameter_constraint = parameter_lhs == parameter_rhs
    parameter_constraint()

    # lets pretend that we optimize the criterion
    network[0].weight.data = torch.zeros_like(network[0].weight.data)
    parameter_constraint()

    # output constraint, output is the output of the combined model (NN + solver)
    @expressionfy
    def output_lhs(output):
        return torch.sum(output["y"])

    @expressionfy
    # output constraint may depend on
    def output_rhs(output):
        return 0.0

    output_constraint = output_lhs == output_rhs
    # neuromancer produces output from input and invokes output
    output_constraint({"y": torch.zeros(1, 10)})
