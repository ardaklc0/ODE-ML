"""
1.1: There five types of DE: ODE(Ordinary), PDE(Partial), SDE(Stochastic), DDE(Delay), and DAE(Differential-Algebraic).
    1.1.1: ODE: ODE is a differential equation that contains one independent variable and one or more dependent variables.
    1.1.2: PDE: PDE is a differential equation that contains two or more independent variables and one or more dependent variables.
    1.1.3: SDE: SDE is a differential equation that contains one independent variable and one or more dependent variables, 
    and the dependent variables are subjected to random noise.
    1.1.4: DDE: DDE is a special type of ODE that contains a time delay.
    1.1.5: DAE: DAE is a generalization of ODE that contains algebraic equations as well as differential equations.

1.2: Typed of Differential Equation Problems: Initial Value Problem(IVP) and Boundary Value Problem(BVP), Dirichlet, Neumann, and Robin boundary conditions.

1.3: Differential Equations Associated with Physical Problems Arising in Engineering
    Coupled L-R electric circuit
    Motion of a pendulum
    Motion of a spring-mass-damper system
    Heat conduction in a rod
    Wave equation
    Laplace's equation

 1.4: General Introduction of Numerical Methods for Solving Differential Equations
 In field of mathematics the existence and uniqueness of the solution of a differential equation is guaranteed by various theorems, but no numerical method for
 obtaining those solutions in explicit and closed form is known. In view of this the limitations of nalytic methods in practical applications have led the 
 evolution of numerical methods for solving differential equations. 
    1.4.1: Shooting Method: The shooting method is a numerical method for solving boundary value problems. It is one of the most popular methods for solving 
    two-point boundary value problems. The shooting method is based on the idea of reducing a boundary value problem to an initial value problem. The solution
    of the initial value problems are then used to approximate (by adding two solutions) the solution of the boundary value problem. And calculated using 
    Runge-Kutta method(4th, 5th, 6th).
    1.4.2: Finite Difference Method: The finite difference method is a numerical method for solving differential equations. Functions are represented by their
    values at a finite number of points. It is an iterative method.
    1.4.3: Finite Element Method: The finite element method is a numerical method for solving differential equations. It is more general than the finite difference 
    method and more useful for real world problems. It is based on the idea of approximating the solution of a differential equation by a piecewise polynomial
    1.4.4: Finite Volume Method
    1.4.5: Spline Based Method
    1.4.6: Neural Network Method: NN can solve both ODE and PDE that relies on the approximation capabilities of feed forward neural networks. It minimizes the
    error between the actual and predicted values of the dependent variable. It requires the computation of the derivatives of the dependent variable with respect,
    which is also called gradient descent method.
"""
