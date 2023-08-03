"""
Ensemble Kalman Filtering. This module is based on the notebook
by Allard de Wit, 2017, available from

https://github.com/ajwdewit/pcse_notebooks/blob/master/08a_data_assimilation_with_the_EnKF.ipynb

under MIT license.

Rewritten and brought into OOP scheme by Lukas Valentin Graf.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes

from temperature_response import Response

variables_for_DA = ['lai']


class EnsembleKalmanFilter:
    """
    Ensemble Kalman filter for assimilating satellite-derived LAI values
    into a temperature response function.

    Parameters
    ----------
    state_vector : pd.DataFrame
        State vector with satellite observed LAI values (observations
        of the true hidden state, i.e., the true LAI).
    n_sim : int
        Number of simulations, i.e., ensemble size.
    Response_calculator : Response
        Temperature response calculator, i.e., the transition function.
    lai_uncertainty : float
        Uncertainty in LAI data (relative).
    process_uncertainty : float
        Process uncertainty (relative) related to the meteorological
        data and the model itself.
    initial_lai_lower_bound : float
        Lower bound for the initial LAI values.
    initial_lai_upper_bound : float
        Upper bound for the initial LAI values.
    new_states : pd.DataFrame
        New states after assimilating the satellite observations.
    """
    def __init__(
            self,
            state_vector: pd.DataFrame,
            n_sim: int,
            response: Response,
            lai_uncertainty: float = 5,
            process_uncertainty: float = 5,
            initial_lai_lower_bound: float = 0.5,
            initial_lai_upper_bound: float = 1.5
    ):
        """
        Class constructor.
        """
        self.state_vector = state_vector
        self.n_sim = n_sim
        self.response = response
        self.lai_uncertainty = lai_uncertainty
        self.process_uncertainty = process_uncertainty
        self.initial_lai_lower_bound = initial_lai_lower_bound
        self.initial_lai_upper_bound = initial_lai_upper_bound
        self.new_states = None

    def _get_analysis_states(
            self,
            lai_value: float,
            lai_states: np.ndarray
    ) -> np.matrix:
        """
        Calculate the update for the ensemble Kalman filter at
        the observation time point.

        Parameters
        ----------
        lai_value : float
            Satellite observed LAI value at the observation time point.
        lai_states : np.ndarray
            Modelled LAI values at the observation time point.
        return : np.matrix
            Update for the ensemble; i.e. the model LAI values
            for the next observation period.
        """
        # using the ensemble Kalman filter
        # get the model stage A
        A_df = pd.DataFrame(lai_states)
        A = np.matrix(A_df)
        # compute the variance within the ensemble A, P_e
        P_e = np.matrix(A_df.T.cov())

        # get the mean and covariance of the satellite observations
        lai_std = self.lai_uncertainty * 0.01 * lai_value
        perturbed_obs = np.random.normal(lai_value, lai_std, (self.n_sim))
        df_perturbed_obs = pd.DataFrame(perturbed_obs)
        df_perturbed_obs.columns = variables_for_DA
        D = np.matrix(df_perturbed_obs).T
        R_e = np.matrix(df_perturbed_obs.cov())

        # Here we compute the Kalman gain
        H = np.identity(len(variables_for_DA))  # len(obs) in the original code
        K1 = P_e * (H.T)
        K2 = (H * P_e) * H.T
        K = K1 * ((K2 + R_e).I)

        # Here we compute the analysed states that will be used to
        # reinitialise the model at the next time step
        Aa = A + K * (D - (H * A))

        return Aa

    def _get_observation_indices(self) -> list[int]:
        """
        Get the indices of the observations in the state vector.
        """
        return [0] + self.state_vector[
                self.state_vector['lai'].notnull()].index.tolist()

    def _get_initial_states(self) -> np.ndarray:
        """
        Get the initial states for the ensemble.
        """
        return np.random.uniform(
            low=self.initial_lai_lower_bound,
            high=self.initial_lai_upper_bound,
            size=(1, self.n_sim))

    def _get_process_uncertainty(
            self,
            lai_states: np.ndarray
    ) -> np.ndarray:
        """
        Get the process uncertainty for the ensemble.

        Parameters
        ----------
        lai_states : np.ndarray
            Modelled LAI values at the observation time point.
        return : np.ndarray
            Process uncertainty for the ensemble.
        """
        return np.random.normal(
            loc=1,
            scale=self.process_uncertainty * 0.01 * np.asanyarray(lai_states),
            size=(1, self.n_sim))

    def _transition_function(
            self,
            time_window: pd.DataFrame,
            lai_value_sim_start: float
    ) -> np.ndarray:
        """
        Transition function simulating LAI development between
        two observations using the temperature response function.

        Parameters
        ----------
        time_window : pd.DataFrame
            Time window for which to simulate LAI development.
        lai_value_sim_start : float
            Initial LAI value.
        return : np.ndarray
            Modelled LAI values.
        """
        response = self.response.get_response(
            time_window['T_mean'])
        response_cumsum = np.cumsum(response) + lai_value_sim_start
        return response_cumsum

    def _transition_function_on_ensembles(
            self,
            time_window: pd.DataFrame,
            lai_value_sim_start: np.ndarray
    ):
        """
        Transition function simulating LAI development between
        two observations using the temperature response function.

        Parameters
        ----------
        time_window : pd.DataFrame
            Time window for which to simulate LAI development.
        lai_value_sim_start : np.ndarray
            Initial LAI values for the ensemble.
        return : tuple(list[np.ndarray], list[float])
            Modelled LAI values for each ensemble member and the
            final LAI values for each ensemble member (i.e., the
            LAI values at the end of the time window for which
            a satellite observation is available)
        """
        lai_modelled = []
        lai_states = []
        for edx, ensemble in enumerate(range(self.n_sim)):
            response_cumsum = self._transition_function(
                time_window,
                lai_value_sim_start[0, edx])
            lai_modelled.append(response_cumsum)
            lai_states.append(response_cumsum[-1])

        return lai_modelled, lai_states

    def run(self) -> None:
        """
        Ensemble Kalman filtering for assimilating satellite-derived
        LAI values into a temperature response function.

        Populates the `new_states` attribute.
        """
        # we need to loop over the available satellite observations
        # and assimilate them one by one
        observation_indices = self._get_observation_indices()

        model_sims_between_points = []
        lai_value_sim_start = None
        Aa = None

        for i in range(len(observation_indices)-1):
            time_window = self.state_vector.loc[
                observation_indices[i]:observation_indices[(i+1)]].copy()

            # set LAI values for initializing the simulations
            # these should reflect typical values at the beginning of
            # the stem elongation phase
            if i == 0:
                # sample from a uniform distribution
                lai_value_sim_start = self._get_initial_states()
            # for the other observations this will be the assimilated
            # LAI value
            else:
                lai_value_sim_start = Aa

            # run the model forward in time for each ensemble member
            # to generate a forecast ensemble
            # this is the transition function
            lai_modelled, lai_states = self._transition_function_on_ensembles(
                time_window,
                lai_value_sim_start)

            # save the model simulations between the satellite observations
            lai_modelled_df = pd.DataFrame(lai_modelled).T
            lai_modelled_df.index = time_window['time']
            model_sims_between_points.append(lai_modelled_df)

            # add process uncertainty to the modelled LAI
            # this could stem from the meteoroogical data
            # or from the model itself
            # we assume that the uncertainty is normally distributed
            # and that it is independent of the model state
            lai_states += self._get_process_uncertainty(lai_states=lai_states)

            # calculate the updates for the next model run
            Aa = self._get_analysis_states(
                lai_value=time_window['lai'].values[-1],
                lai_states=lai_states)

        model_sims_between_points = pd.concat(
            model_sims_between_points, axis=0)
        # update attribute
        self.new_states = model_sims_between_points.copy()

    def plot_new_states(self, ax: Axes = None) -> plt.Figure:
        """
        Plot the new states.

        Returns
        -------
        fig : plt.Figure
            Figure object.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.get_figure()

        ax.plot(
            self.state_vector['time'],
            self.state_vector['lai'],
            'o',
            label='Satellite GLAI',
            color='red'
        )
        for i in range(self.n_sim):
            ax.plot(self.new_states.index, self.new_states[i])
        ax.set_xlabel('Time')
        ax.set_ylabel(r'GLAI [m$^2$ m$^{-2}$]')
        ax.legend()
        return fig
