import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jaxns import (
    Model,
    NestedSampler,
    Prior,
    plot_cornerplot,
    plot_diagnostics,
    summary,
)
from starccato_jax import StarccatoVAE

tfpd = tfp.distributions


# Define model setup
def create_model(MODEL, y_obs):
    latent_dim = MODEL.latent_dim

    def prior_model():
        # Define normal priors for each latent dimension
        z_parts = []
        for i in range(latent_dim):
            z_i = yield Prior(tfpd.Normal(loc=0.0, scale=1.0), name=f"z_{i}")
            z_parts.append(z_i)
        z = jnp.stack(z_parts, axis=0)

        # Generate model prediction from latent variables
        model_pred = MODEL.generate(z=z)
        return model_pred

    def log_likelihood(model_pred):
        # Gaussian likelihood with fixed sigma=0.1
        inv_sigma2 = 1.0 / (0.1**2)
        return -0.5 * jnp.sum(
            (y_obs - model_pred) ** 2 * inv_sigma2 - jnp.log(inv_sigma2)
        )

    return Model(prior_model=prior_model, log_likelihood=log_likelihood)


# Generate mock data (as in your example)
def generate_data(MODEL, rng_key):
    latent_dim = MODEL.latent_dim
    z_true = jax.random.normal(rng_key, shape=(latent_dim,))
    y_true = MODEL.generate(z=z_true)

    # Add Gaussian noise
    y_err = 0.1 * jax.random.normal(rng_key, shape=y_true.shape)
    y_obs = y_true + y_err

    return z_true, y_obs


def main():
    # Initialize model and RNG
    rng_key = jax.random.PRNGKey(0)
    MODEL = StarccatoVAE()  # Your model class

    # Generate mock data
    z_true, y_obs = generate_data(MODEL, rng_key)

    # Create JAXNS model
    model = create_model(MODEL, y_obs)

    # Sanity check the model
    model.sanity_check(rng_key, S=10)

    # Setup nested sampler
    ns = NestedSampler(
        model=model,
        max_samples=1e4,
        k=50,  # Number of live points
        c=50,  # Number of MCMC steps
        s=50,  # Number of slices
    )

    # Run sampling
    termination_reason, state = ns(rng_key)
    results = ns.to_results(termination_reason=termination_reason, state=state)

    # Analysis and plotting
    summary(results)
    plot_diagnostics(results)

    # # Corner plot with true values
    # param_names = [f"z_{i}" for i in range(MODEL.latent_dim)]
    # plot_cornerplot(results, save_name="cornerplot.png")


if __name__ == "__main__":
    main()
