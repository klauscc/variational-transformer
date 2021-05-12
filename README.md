# Time Series Modeling via a Variational Transformer

## Prerequisites

Install `torchdiffeq` from https://github.com/rtqichen/torchdiffeq.

To run the MuJoCo Physical Simulation experiment, [DeepMind Control Suite](https://github.com/deepmind/dm_control/) is required

## Experiments on Bi-directional Spiral datasets

*  to be added
```
to be added
```

## Experiments on the MuJoCo Physical Simulation

* run the latent-ODE model with RNN encoder
```
python3 run_models.py --niters 300 -n 1000 -l 15 --dataset hopper --latent-ode --z0-encoder rnn
```

* run the latent-ODE model with Transformer encoder 
```
python3 run_models.py --niters 300 -n 1000 -l 15 --dataset hopper --latent-ode --z0-encoder trans

```

* run the latent-ODE model with Transformer encoder for 30% observed time points
```
python3 run_models.py --niters 300 -n 1000 -l 15 --dataset hopper --latent-ode --z0-encoder trans -s 0.3

```
