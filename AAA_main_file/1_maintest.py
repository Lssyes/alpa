from typing import Any
import alpa
from alpa.testing import assert_allclose
import copy
from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from jax import random
import optax
import ray
from alpa.util import benchmark_func
alpa.util.disable_tqdm_globally()
a = alpa.init(cluster="ray")




class MaunulModelPipeline(nn.Module):
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        # x = nn.Dense(features=self.hidden_dim)(x)
        # x = nn.relu(x)
        # x = nn.Dense(features=self.hidden_dim)(x)
        # x = nn.relu(x)
        # x = nn.Dense(features=self.hidden_dim)(x)
        # x = nn.Dense(features=self.hidden_dim)(x)
        # x = nn.Dense(features=self.hidden_dim)(x)
        # x = nn.relu(x)
        # x = nn.Dense(features=self.hidden_dim)(x)
        # x = nn.relu(x)
        return x



dim = 8
batch_size = 128

# Generate ground truth W and b
rngkey = jax.random.PRNGKey(0)
k1, k2 = random.split(rngkey)
W = random.normal(k1, (dim, dim))
b = random.normal(k2, (dim,))

# Generate the training data
ksample, knoise = random.split(k1)
x = random.normal(ksample, (batch_size, dim))
y = (x @ W + b) + 0.1 * random.normal(knoise, (batch_size, dim))

# Initialize a train state, which includes the model paramter and optimizer
# state.
model = MaunulModelPipeline(hidden_dim=dim)
params = model.init(rngkey, x)
tx = optax.adam(learning_rate=1e-3)
tx = optax.sgd(learning_rate=1e-3)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# print(params)
print(jax.tree_util.tree_map(lambda x: x.shape, params))




@alpa.parallelize(method=alpa.PipeshardParallel(num_micro_batches=16,
                                                layer_option=alpa.AutoLayerOption(layer_num=2),
                                                stage_option="auto"))
def MaunulModelPipeline_TrainStep(state, batch):
    def loss(params):
        out = state.apply_fn(params, batch["x"])
        loss = jnp.mean((batch["y"] - out)**2)
        return loss
    grads = alpa.grad(loss)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state


import time
batch = {"x": x, "y": y}

# for i in range(100):
#     start_time = time.time()
#     state = MaunulModelPipeline_TrainStep(state, batch)
#     a = state.params
#     tiktime = time.time()
#     print(f"step-{i}: {tiktime-start_time}")
import numpy as np
def alpa_execution():
    global state
    state = MaunulModelPipeline_TrainStep(state, batch)
def sync_func():
    # jax.local_devices()[0].synchronize_all_activity()
    jax.device_put(0.).block_until_ready()

alpa_costs = benchmark_func(alpa_execution,  warmup=5, number=10, repeat=5) * 1e3
print(f"Alpa execution time.   Mean: {np.mean(alpa_costs):.2f} ms, Std: {np.std(alpa_costs):.2f} ms")

alpa.shutdown()
