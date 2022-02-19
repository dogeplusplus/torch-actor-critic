# torch-actor-critic

`torch-actor-critic` is a PyTorch implementation of Soft Actor-Critic. The implementation is cross-referenced against popular implementations such as OpenAI's `spinningup`.

Additional features include:
- Mlflow logging of parameters and runs for easier experiment evaluation and comparison
- Create multiple concurrent environments and synchronise model weights using MPI across separate threads
- Gym wrapper for `dm_control` environments

![torch-actor-critic demonstration](https://github.com/dogeplusplus/torch-actor-critic/blob/main/assets/mujoco_demo.gif)
