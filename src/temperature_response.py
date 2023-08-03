"""
Class for calculating the temperature response from
environmental covariates.

@author: Flavian Tschurr
"""

import numpy as np


class Response:
    def __init__(self, response_curve_type, response_curve_parameters):
        self.response_cruve_type = response_curve_type
        self.params = response_curve_parameters

    def non_linear_response(self, env_variate):
        '''
        env_variate: value of an environmental covariate
        base_value: estimated value, start of the linear growing phase
        slope: estimated value, slope of the linear phase
        description: broken stick model according to an env variable
        '''

        base_value = self.params.get('base_value', 0)
        slope = self.params.get('slope_value', 0)

        y = (env_variate - base_value) * slope
        y = y if env_variate > base_value else 0.
        return y

    def asymptotic_response(self, env_variate):
        """
        Calculates the asymptotic response for a given input variable.

        Args:
        env_variate: input variable
        Asym: a numeric parameter representing the horizontal asymptote on
        the right side (very large values of input).
        lrc: a numeric parameter representing the natural logarithm of the
        rate constant.
        c0: a numeric parameter representing the env_variate for which the
        response is zero.

        Returns:
        A numpy array containing the asymptotic response values.
        """
        Asym = self.params.get('Asym_value', 0)
        lrc = self.params.get('lrc_value', 0)
        c0 = self.params.get('c0_value', 0)

        y = Asym * (1. - np.exp(-np.exp(lrc) * (env_variate - c0)))
        y = np.where(y > 0., y, 0.)  # no negative growth
        return y

    def wangengels_response(self, env_variate):
        """
        Calculates the Wang-Engels response for a given input variable.

        Args:
            env_variate: effective env_variable value

        Returns:
            A numpy array containing the Wang-Engels response values.
        """
        xmin = self.params['xmin_value']
        xopt = self.params['xopt_value']
        xmax = self.params['xmax_value']

        alpha = np.log(2.) / np.log((xmax - xmin) / (xopt - xmin))

        if xmin <= env_variate <= xmax:
            y = (2. * (env_variate - xmin) ** alpha *
                 (xopt - xmin) ** alpha - (env_variate - xmin) **
                 (2. * alpha)) / \
                ((xopt - xmin) ** (2. * alpha))
        else:
            y = 0

        return y

    def get_response(self, env_variates):
        response_fun = getattr(
            Response, f'{self.response_cruve_type.lower()}_response')
        response = []
        for env_variate in env_variates:
            response.append(response_fun(self, env_variate))
        return response