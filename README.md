# TMAP
Code for the paper  TMAP: Integrating Trust Region and Maximum Entropy with Augmented Bellman Equation for Policy Optimization

This code is based on [openai spinning up](https://github.com/openai/spinningup)

## Requirements

We assume you have access to a gpu that can run CUDA. All of the dependencies are in the `setup.py` file.

```
pip install -e .
```

After the installation ends you can run the code.



## Instructions

To train a TMAP agent on the `Ant-v3` task, you can first change your path to `the_path_of_TMAP/spinup/algo/tmap/` and then run the following code

```
CUDA_VISIBLE_DEVICES=0 python tmap.py --md 0.08 --epoch 750 --env Ant-v3
```

