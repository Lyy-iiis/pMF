pip install jax[tpu]==0.4.34 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install jaxlib==0.4.34 "flax>=0.8"
pip install pillow clu tensorflow==2.15.0 "keras<3" "torch<=2.4" torchvision
pip install orbax-checkpoint==0.6.4 ml-dtypes==0.5.4 tensorstore==0.1.78
pip install wandb lpips_j optax ml-collections