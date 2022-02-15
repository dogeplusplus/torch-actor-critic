# torch-actor-critic

`torch-actor-critic` is a PyTorch implementation of Soft Actor-Critic. The implementation is cross-referenced against popular implementations such as OpenAI's `spinningup`.

Additional features include:
- Option to use Welford's Method for variance estimation. This is used to normalize the state prior to doing inference
- Mlflow logging of parameters and runs for easier experiment evaluation and comparison
- Create multiple concurrent environments and synchronise model weights using MPI across separate threads

![torch-actor-critic demonstration](https://github.com/dogeplusplus/torch-actor-critic/blob/main/assets/mujoco_demo.gif)
