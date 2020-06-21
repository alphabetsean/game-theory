import pandas as pd
import numpy as np

# This table summarizes the assumptions on severity which are informed by epidemiological and clinical observations in China. 
# The first column reflects our assumption on what fraction of infections are reflected in the statistics from China, 
# the following columns contain the assumption on what fraction of the previous category deteriorates to the next. 

# heavily based on https://github.com/neherlab/covid19_scenarios

covid19_severity_assumptions=pd.DataFrame({
    "age":["0-9","10-19","20-29","30-39","40-49","50-59","60-69","70-79","80+"],
    "pct_confirmed":[.05,.05,.1,.15,.2,.25,.3,.4,.5],
    "pct_severity":[.01,.03,.03,.03,.06,.1,.25,.35,.5],
    "pct_critical":[.05,.1,.1,.15,.2,.25,.35,.45,.55],
    "pct_fatal":[.3,.3,.3,.3,.3,.4,.4,.5,.5]
})

covid19_population_params_germany={
    "population":83784000,
    "age_distribution":np.array([.093,.096,.112,.128,.125,.162,.124,.091,.069]),
    "initial_days":14,
    "imports_per_day":12.2,
    "hospital_beds":500680,
    "icu_beds":23890,
    "isolation_effectiveness":.9
}

covid19_epidemiology_params={
    "R0":2.7,
    "latency_period":5,
    "infectious_period":3,
    "hospital_stay":4,
    "icu_stay":14,
    "icu_overflow_severity":2
}