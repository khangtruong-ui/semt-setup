pip install -q jax[tpu] flax optax datasets Pillow matplotlib einops eqxvision equinox grain -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

gcloud storage cp -r $DATASET_DIRECTORY .
