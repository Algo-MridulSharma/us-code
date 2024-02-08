"""
Contact: psar@algoanalytics.com
Company: AlgoAnalytics Pvt. Ltd.
"""
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize

def portfolio_optim(covariance_matrix,num_assets,total_assets):
    """
    This function performs portfolio optimization by minimizing the portfolio variance.
    optimisation_function -> x' * covariance * x
    Parameters:
        covariance_matrix (np.ndarray): The covariance matrix of asset returns.
        num_assets (int): The number of assets to be included in the portfolio.
        total_assets (int): The total number of assets available.

    Returns:
        np.ndarray: The optimal allocation of assets in the portfolio.

    """

    # Define the decision variables
    x = cp.Variable(total_assets, boolean=True)
    mod_x = x/num_assets #Summation of weights of assets should always be 1


    # quad_form is quad_form(x,P) <-> x'Px -> total portfolio variance
    # Define the objective function (minimize portfolio variance)
    objective = cp.Minimize(cp.quad_form(mod_x, covariance_matrix))

    # Define the constraints
    constraints = [
        cp.sum(x) == num_assets,
        x >= 0
    ]

    # Create the optimization problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()

    # Get the optimal solution
    optimal_allocation = x.value
    print("Optimal Allocation:", optimal_allocation)
    return optimal_allocation

def portfolio_return_max_with_variance(nifty_etf_variance, 
                                       expected_returns,
                                       covariance_matrix,
                                       num_assets,
                                       total_assets,type):
    """
    This function performs portfolio optimization by maximizing the portfolio return.
    Constraint: 
    optimisation_function -> r * x (r = expected returns)
    Parameters:
        covariance_matrix (np.ndarray): The covariance matrix of asset returns.
        num_assets (int): The number of assets to be included in the portfolio.
        total_assets (int): The total number of assets available.

    Returns:
        np.ndarray: The optimal allocation of assets in the portfolio.

    """

    # Define the decision variables
    x = cp.Variable(total_assets, boolean=True)
    mod_x = x/num_assets #Summation of weights of assets should always be 1

    # quad_form is quad_form(x,P) <-> x'Px

    # Define the objective function (maximize portfolio return)
    portfolio_return = expected_returns @ mod_x
    portfolio_risk = cp.quad_form(mod_x, covariance_matrix)
    sharpe_ratio = (portfolio_return) / cp.sqrt(portfolio_risk) 
    objective = cp.Maximize(portfolio_risk)

    # Define the constraints
    if type == "Constrained":
        constraints = [
            cp.sum(x) == num_assets,
            x >= 0,
            portfolio_risk <= nifty_etf_variance 
        ]
    elif type == "UnConstrained":
        constraints = [
            cp.sum(x) == num_assets,
            x >= 0
        ]


    # Create the optimization problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()

    # Get the optimal solution
    optimal_allocation = x.value
    print("Optimal Allocation:", 
          optimal_allocation, 
          portfolio_risk.value, 
          nifty_etf_variance)
    return optimal_allocation

def portfolio_sharpe_max_iterative_scipy(expected_returns,
                                       covariance_matrix,
                                       num_assets,
                                       total_assets,
                                       top_n = True):
    """_summary_

    Args:
        expected_returns (_type_): _description_
        covariance_matrix (_type_): _description_
        num_assets (_type_): _description_
        total_assets (_type_): _description_
        top_n (bool, optional):Defaults to True. 
            If top N is "True" then top 8 stocks ...
            with highest weightage are selected
            IF top N is "False" then weights are rounded off ...
            to nearest integer

    Returns:
      

    """

    def negative_sharpe_ratio(weights,num_assets, 
                              expected_returns,
                              covariance_matrix):
        mod_weights = weights/num_assets
        portfolio_return = np.dot(expected_returns, mod_weights)
        portfolio_stddev = np.sqrt(np.dot(mod_weights.T, np.dot(
            covariance_matrix, 
            mod_weights)))
        sharpe_ratio = (portfolio_return ) / portfolio_stddev
        return -sharpe_ratio

    constraints = (
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - num_assets},
        {'type': 'ineq', 'fun': lambda weights: weights}
        )

    bounds = [(0,1)] * total_assets
    initial_guess = np.random.choice([0, 1], size=total_assets)
    result = minimize(negative_sharpe_ratio,
                    initial_guess,
                    args=(num_assets,expected_returns,covariance_matrix),
                    bounds=bounds,
                    constraints=constraints)
    weighted_allocation = result.x

    # Exception Case: Optimisation Failed
    if np.all(weighted_allocation == 0.0): 
        raise Exception("Optimisation failed!")
    
    if top_n: #method to use top n assets
        sorted_weight_allocation = np.sort(weighted_allocation)
        threshold_weight = sorted_weight_allocation[-1*num_assets]
        optimal_allocation = weighted_allocation.copy()
        optimal_allocation[optimal_allocation >= threshold_weight] = 1
        optimal_allocation[optimal_allocation < threshold_weight] = 0
    else: #method to use rounded off weights as selected assets
        optimal_allocation = np.round(result.x)
    print(result)
    return optimal_allocation



if __name__ == "__main__":
    # Simple Test case
    expected_returns = np.array([0.00026028, 0.00346769, 0.00775207])
    
    covariance_matrix = np.array([[0.0070449 , 0.00103099, 0.00086981],       
                                [0.00103099, 0.00274846, 0.00119827],       
                                [0.00086981, 0.00119827, 0.00177477]])
    num_assets = 2
    total_assets = 3
    optimal_allocation = portfolio_return_max_with_variance(0.01,expected_returns,covariance_matrix, num_assets, total_assets)
    print("Test Case - Optimal Allocation:", optimal_allocation)




