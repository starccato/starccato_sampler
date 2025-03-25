import jax
import tensorflow_probability.substrates.jax as tfp
from jaxns import (
    NestedSampler,
    load_results,
    plot_cornerplot,
    plot_diagnostics,
    save_results,
    summary,
)
from jaxns.framework.model import Model
from jaxns.framework.prior import Prior
from starccato_jax import StarccatoVAE

tfpd = tfp.distributions


def prior_model():
    mu = yield Prior(tfpd.Normal(loc=0.0, scale=1.0))
    # Let's make sigma a parameterised variable
    sigma = yield Prior(
        tfpd.Exponential(rate=1.0), name="sigma"
    ).parametrised()
    x = yield Prior(tfpd.Cauchy(loc=mu, scale=sigma), name="x")
    uncert = yield Prior(tfpd.Exponential(rate=1.0), name="uncert")
    return x, uncert


def log_likelihood(x, uncert):
    return tfpd.Normal(loc=0.0, scale=uncert).log_prob(x)


model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

# You can sanity check the model (always a good idea when exploring)
model.sanity_check(key=jax.random.PRNGKey(0), S=100)

# The size of the Bayesian part of the prior space is `model.U_ndims`.

# Sample the prior in U-space (base measure)
U = model.sample_U(key=jax.random.PRNGKey(0))
# Transform to X-space
X = model.transform(U=U)
# Only named Bayesian prior variables are returned, the rest are treated as hidden variables.
assert set(X.keys()) == {"x", "uncert"}

# Get the return value of the prior model, i.e. the input to the likelihood
x_sample, uncert_sample = model.prepare_input(U=U)

# Evaluate different parts of the model
log_prob_prior = model.log_prob_prior(U)
log_prob_likelihood = model.log_prob_likelihood(U, allow_nan=False)
log_prob_joint = model.log_prob_joint(U, allow_nan=False)

init_params = model.params


def log_prob_joint_fn(params, U):
    # Calling model with params returns a new model with the params set
    return model(params).log_prob_joint(U, allow_nan=False)


value, grad = jax.value_and_grad(log_prob_joint_fn)(init_params, U)


ns = NestedSampler(model=model, max_samples=1e5)

# Ahead of time compilation (sometimes useful)
ns_aot = jax.jit(ns).lower(jax.random.PRNGKey(42)).compile()

# Just-in-time compilation (usually useful)
ns_jit = jax.jit(ns)

# Run the sampler
termination_reason, state = ns(jax.random.PRNGKey(42))
# Get the results
results = ns.to_results(termination_reason=termination_reason, state=state)

# Optionally save the results to file
save_results(results, "results.json")
# To load the results back use this
results = load_results("results.json")

summary(results)
plot_diagnostics(results)
plot_cornerplot(results)
