from .config import covid19_epidemiology_params, covid19_population_params_germany, covid19_severity_assumptions
from .render import StockTradingGraph
import gym
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000

LOOKBACK_WINDOW_SIZE = 40

class Covid19Gym(gym.Env):
    metadata = {'render.modes': ['live', 'file', 'none']}
    visualization = None
    def __init__(
            self, population_params=covid19_population_params_germany,
            epidemiology_params=covid19_epidemiology_params,
            severity_assumptions=covid19_severity_assumptions, time_delta=1,
            max_steps=150):
        super().__init__()
        self.population_params = population_params
        self.epidemiology_params = epidemiology_params
        self.severity_assumptions = severity_assumptions
        self.time_delta = time_delta
        self.max_steps = max_steps


        # state is a dict of different age group where each age group is a dict
        # of different population
        state = self.reset()

        self.observation_space = gym.spaces.Box(0, 1, state.shape)
        self.action_space = gym.spaces.Box(0, 1, (1,))

    def calc_state(self):
        """Calculate population structure.
        """
        values = []
        for age in self.state:
            for k in self.state[age]:
                if k not in ["susceptible", "exposed"]:
                    values.append(self.state[age][k])
        values = np.array(values).astype(np.float)
        values /= self.population_params["population"]
        return values.reshape(-1)

    def reset(self):
        """Reset the environment starting with one infectious individuals.  


        Assuming starting with with one initial cases, the 'exposed' ones were 
        calcualted as the initial cases times the age distribution and taken as
        integer. 'susceptible' is the population of this age range minus 
        'exposed' ones.

        The environment run for 14 days with no isolation.
        """

        inital_cases = 1
        initial_steps = np.random.poisson(
            self.population_params["initial_days"])

        self.state = {}
        for i, d in self.severity_assumptions.iterrows():
            age_state = {
                "susceptible": 0,
                "exposed": 0,
                "infectious": 0,
                "severe": 0,
                "critical": 0,
                "fatal": 0,
                "recovered": 0
            }
            age_state["exposed"] = int(
                inital_cases * self.population_params["age_distribution"][i])
            age_state["susceptible"] = int(self.population_params['population'] *
                                           self.population_params["age_distribution"][i]) - age_state['exposed']
            self.state[d.age] = age_state

        for _ in range(initial_steps):
            self.steps = 0
            self.step([0., 0.])

        self.steps = 0

        return self.calc_state()

    def step(self, action):
        """Take action 

        Parameters
        ----------
        action : float
            a number in [0, 1] to measure the isolation policy implemented in 
            each step
        """

        # isolation fraction: how much of an population is isolated
        isolation = min(1., max(0., action[0])) * self.population_params["isolation_effectiveness"]

        # calculated the total number of a category, e.g. 'fatal'
        def total(k): return np.array([self.state[age][k] for age in self.state]).sum()

        # randomly draw a poisson number
        def sample(x): return np.random.poisson(x)

        # calculate fraction of infected
        total_infected = total("exposed") + total("infectious")
        frac_infected = total_infected / self.population_params["population"]

        ###############################
        # Paramemters for progression #
        ###############################

        # the basic idea is get the average as the poisson distribution for
        # calculation later

        # average infection rate: R0 / (infectious_period + latency_period)
        avg_infection_rate = self.epidemiology_params["R0"] / (self.epidemiology_params["infectious_period"] + self.epidemiology_params["latency_period"])

        # average latency rate
        latency_rate = 1 / self.epidemiology_params["latency_period"]

        # average infectious rate
        infectious_rate = 1 / self.epidemiology_params["infectious_period"]
        
        # average hospital rate
        hospital_rate = 1 / self.epidemiology_params["hospital_stay"]
        
        # average icu rate
        icu_rate = 1 / self.epidemiology_params["icu_stay"]


        ##########
        # REWARD #
        ##########
        
        # TODO: More complex strategies are conceivable
        isolated_frac = isolation

        ####################
        # Population shift #
        ####################
        for (i, age) in enumerate(self.state):
            sa = self.severity_assumptions.iloc[i]

            # Susceptible -> Exposed
            dedt = min(
                sample(
                    self.population_params["imports_per_day"] *
                    sa.pct_confirmed * self.time_delta * (1 - isolation)
                ) + sample(
                    self.state[age]["susceptible"] * frac_infected *
                    avg_infection_rate * self.time_delta * (1 - isolation)
                ),
                self.state[age]["susceptible"]
            )

            # Exposed->Infectious
            didt = min(
                sample(
                    latency_rate * self.state[age]["exposed"] * self.time_delta
                ),
                self.state[age]["exposed"]
            )

            # Infectious->Recovered/Severe
            drdt = min(
                sample(
                    infectious_rate * (1 - sa.pct_severity) *
                    self.state[age]["infectious"] * self.time_delta
                ),
                self.state[age]["infectious"]
            )
            dsdt = min(
                sample(
                    infectious_rate *
                    (sa.pct_severity) *
                    self.state[age]["infectious"] * self.time_delta
                ),
                self.state[age]["infectious"] - drdt
            )

            # Severe->Recovered/Critical
            drdt_from_severe = min(
                sample(
                    self.state[age]["severe"] *
                    (1 - sa.pct_critical) * hospital_rate * self.time_delta
                ),
                self.state[age]["severe"]
            )
            dcdt = min(
                sample(
                    self.state[age]["severe"] * sa.pct_critical *
                    hospital_rate * self.time_delta
                ),
                self.state[age]["severe"] - drdt_from_severe
            )

            # Critical -> Fatal/Severe
            dsdt_from_critcal = min(
                sample(
                    self.state[age]["critical"] *
                    (1 - sa.pct_fatal) * icu_rate * self.time_delta
                ),
                self.state[age]["critical"]
            )
            dfdt = min(
                sample(
                    self.state[age]["critical"] *
                    sa.pct_fatal * icu_rate * self.time_delta
                ),
                self.state[age]["critical"] - dsdt_from_critcal
            )

            # update state
            self.state[age]["susceptible"] -= dedt
            self.state[age]["exposed"] += dedt
            self.state[age]["exposed"] -= didt
            self.state[age]["infectious"] += didt
            self.state[age]["recovered"] += drdt
            self.state[age]["infectious"] -= drdt
            self.state[age]["severe"] += dsdt
            self.state[age]["infectious"] -= dsdt
            self.state[age]["recovered"] += drdt_from_severe
            self.state[age]["severe"] -= drdt_from_severe
            self.state[age]["critical"] += dcdt
            self.state[age]["severe"] -= dcdt
            self.state[age]["severe"] += dsdt_from_critcal
            self.state[age]["critical"] -= dsdt_from_critcal
            self.state[age]["fatal"] += dfdt
            self.state[age]["critical"] -= dfdt

        self.steps += 1

        # finish for fixed amount of time unit
        done = self.steps >= self.max_steps

        # TODO: the reward of such is too simple and does not incorporate cost
        # of central planning
        reward = - isolated_frac

        # finish if no icu beds are aviable
        if total("critical") > self.population_params["icu_beds"]:
            done = True
            reward -= 1000

        return self.calc_state(), reward, done, {}

    def _render_to_file(self, filename='render.txt'):

        def total(k): return np.array([self.state[age][k] for age in self.state]).sum()

        file = open(filename, 'a+')

        file.write(f'Step: {self.steps}\n')
        file.write(f'Balance: 100\n')
        file.close()

    def render(self, mode='live', **kwargs):
        # Render the environment to the screen
        if mode == 'file':
            self._render_to_file(kwargs.get('filename', 'render.txt'))

        elif mode == 'live':
            if self.visualization == None:
                self.visualization = StockTradingGraph(self.state, kwargs.get('title', None))

            # if self.current_step > LOOKBACK_WINDOW_SIZE:
            #     self.visualization.render(
            #         self.current_step, self.net_worth, self.trades, window_size=LOOKBACK_WINDOW_SIZE)

            if self.steps > LOOKBACK_WINDOW_SIZE:
                self.visualization.render(
                    self.steps, 100, 2312, window_size=LOOKBACK_WINDOW_SIZE)
    
    def close(self):
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None