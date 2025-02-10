"""Script to find coefficients for an ODE solution for the motion of the puck on the air table."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp
import analyze_puck_data as analyze_data


PROJECT_PATH = str(Path(__file__).resolve().parents[0])


# Solving an ODE of the form x'' = -f/m -d/m (x')^2, x is position, m is mass, f is kinetic friction, d is air resistance
#first combine f/m, d/m into single constants a, b
#ODE becomes x'' = -a - b(x')^2
#plug into wolfram alpha to get x(t) and x'(t)


class CurveFitter:
    """Class to fit the data to a curve and solve for the ODE parameters"""

    def __init__(self, datacsvpath):
        self.datacsvpath = datacsvpath
        self.sa, self.sb, self.sc1, self.sc2, self.st = sp.symbols('a b c_1 c_2 t')
        self.datasegments = analyze_data.segment_data(datacsvpath)
        self.a = 0
        self.b = 0

        self.x_turbulent_sp = sp.log(sp.cos(sp.sqrt(self.sa)*sp.sqrt(self.sb)*(self.st+self.sc1)))/self.sb + self.sc2
        self.x_prime_turbulent_sp = -self.sa*sp.tan(sp.sqrt(self.sa)*sp.sqrt(self.sb)*(self.st+self.sc1))/sp.sqrt(self.sa*self.sb)
        self.x_laminar_sp = -self.sa*self.st/self.sb + self.sc1*sp.exp(-self.sb*self.st)/self.sb + self.sc2
        self.x_prime_laminar_sp = self.sa/self.sb - self.sc1*sp.exp(-self.sb*self.st)


    def solve_ICs(self, collision_num, equation_type='turbulent'):
        """Fit the data to the ODE solution"""
        if equation_type == 'turbulent':
            x = self.x_turbulent_sp
            x_prime = self.x_prime_turbulent_sp
        else:
            x = self.x_laminar_sp
            x_prime = self.x_prime_laminar_sp

        # Select a single segment (e.g., segment 1)
        segment = self.datasegments[self.datasegments['collision'] == collision_num]

        #the initial values right after the collision
        x0 = round(segment['y'].iloc[0],3)
        print(segment['vy'].iloc[0])
        print(segment['y'].iloc[0])
        x_prime0 = round(segment['vy'].iloc[0],3)
        # x_prime0 = 0

        eq1 = sp.Eq(x.subs(self.st, 0), x0)      # x(0) = x0
        eq2 = sp.Eq(x_prime.subs(self.st, 0), x_prime0) # x'(0) = x'0
        print(eq1)
        print(eq2)
        # Solve for c1 and c2
        if equation_type == 'turbulent':
            solution = sp.solve((eq1, eq2), (self.sc1, self.sc2))[0]
            sc1_sol = solution[0]
            sc2_sol = solution[1]
        else :
            solution = sp.solve((eq1, eq2), (self.sc1, self.sc2))
            sc1_sol = solution[self.sc1]
            sc2_sol = solution[self.sc2]
        c1_func = sp.lambdify((self.sa, self.sb), sc1_sol, 'numpy')
        c2_func = sp.lambdify((self.sa, self.sb), sc2_sol, 'numpy')
        return c1_func, c2_func

    def fit_curve(self, collision_num, equation_type='turbulent'):
        """Fit the data to a single segment of puck motion
        
        :param collision_num: the segment number to fit
        :param equation_type: the type of ODE to fit to, either 'turbulent' or 'laminar'
        :return: the fitted parameters a and b"""
        c1_func, c2_func = self.solve_ICs(collision_num, equation_type)
        # Define the function for x(t) to fit, using the solved c1 and c2
        if equation_type == 'turbulent':
            def x_func(t, a, b):
                c1_value = c1_func(a, b)
                c2_value = c2_func(a, b)
                return np.log(np.cos(np.sqrt(a)*np.sqrt(b)*(t+c1_value)))/b + c2_value
        else:
            def x_func(t, a, b):
                c1_value = c1_func(a, b)
                c2_value = c2_func(a, b)
                return -a*t/b + c1_value*np.exp(-b*t)/b + c2_value

        # Prepare the data for fitting
        segment = self.datasegments[self.datasegments['collision'] == collision_num]
        t_data = segment['dt'].cumsum().values
        x_data = segment['y'].values

        # Initial guess for the parameters [a, b]
        initial_guess = [0.1, 1]

        # Fit the function to the data
        params, _ = curve_fit(x_func, t_data, x_data, p0=initial_guess)
        self.a = params[0]
        self.b = params[1]

        # Print the fitted parameters
        return params
    
    def plot_curve(self, collision_num, a_val=None, b_val=None, equation_type='turbulent'):
        """Plot the data and the fitted curve for a single segment of puck motion
        
        :param collision_num: the segment number to plot
        :param a_val: the value of the parameter a to use in the fitted curve
        :param b_val: the value of the parameter b to use in the fitted curve
        :param equation_type: the type of ODE to fit to, either 'turbulent' or 'laminar'"""
        if a_val is None:
            a_val = self.a
        if b_val is None:
            b_val = self.b
        c1_func, c2_func = self.solve_ICs(collision_num, equation_type)
        
        if equation_type == 'turbulent':
            def x_func(t, a, b):
                c1_value = c1_func(a, b)
                c2_value = c2_func(a, b)
                return np.log(np.cos(np.sqrt(a)*np.sqrt(b)*(t+c1_value)))/b + c2_value
        else:
            def x_func(t, a, b):
                c1_value = c1_func(a, b)
                c2_value = c2_func(a, b)
                return -a*t/b + c1_value*np.exp(-b*t)/b + c2_value

        #plot the data and the fitted curve
        data = analyze_data.segment_data(self.datacsvpath)
        segment = data[data['collision'] == collision_num]
        t_data = segment['dt'].cumsum().values
        x_data = segment['y'].values
        plt.plot(t_data, x_data, label='Data')
        plt.plot(t_data, x_func(t_data, a_val, b_val), label='Fitted function')
        plt.legend(loc='best')
        plt.show()


if __name__ == "__main__":
    #plot the data and the fitted curve
    curve = CurveFitter(PROJECT_PATH + '/data/position_13.csv')
    a_val, b_val = curve.fit_curve(7, equation_type='laminar')
    curve.plot_curve(5, a_val=a_val, b_val=b_val, equation_type='laminar')
    #curve.plot_curve(9, 1, 0.3)