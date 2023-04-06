# MARLMOT
Multi-Agent Reinforcement Learning for Multiple Object Tracking

This respository contains an implementation of [MARLMOT](https://ifaamas.org/Proceedings/aamas2018/pdfs/p1397.pdf) which is a multiple object tracking algorithm that extends [SORT](https://arxiv.org/pdf/1602.00763.pdf) by using multiple agents to manage each track. The agents decide how to update track filters and when to delete, keep, or mark as hidden. Each agent is parameterized as a Feed Forward Neural Network that observes continuous data from a track and outputs probabilities for 5 discrete actions that determine how each track is updated

<p align="center">
  <img src="https://user-images.githubusercontent.com/60835780/229327407-9b70cb6a-1ef4-4eec-8b38-c9e06ffd26c0.png" width=50% height=50%>
</p>


Since MARLMOT is a simple model that directly observes each track it is able to be modularly implemented in a tracking pipeline


<p align="center">
  <img src="https://user-images.githubusercontent.com/60835780/229327550-1b236580-7abd-4369-8d9e-4fcd6c85cb3c.png" width=75% height=75%>
</p>


## Implementation 
This implementation is slightly different than the original, see [this](https://medium.com/@itberrios6/marlmot-4f018282e0cc) for more information about the original implementation. 

The Joint combinations of actions taken by N agents is intractable, so each agent is parameterized by the same policy. The rewards are computed every frame and are based off of multiple object tracking metrics. During training, the same policy is used to sample actions for each observation and at each frame the same rewards are given to each agent. In this manner all of the agents are incentivized to cooporatively track all targets.


## Training
The original model was trained with [MOT15](https://motchallenge.net/data/MOT15/) data using Trust Region Policy Optimization, this version uses Proximal Policy Optimization. To train the model from scratch either specify the paths to the MOT15 train folder and savepaths in the train.py file or pass them as arguments. <br>
```train.py``` <br>
``` train.py --trainfolder path\to\mot15\trainfolder --savepath path\to\save\models ```


## Evaluation
To evaluate a trained policy/actor model and print metrics, run "eval.py". Update argparser to set desired default arguments and/or pass them in via commandline. This mode requires that truth data be available and currently works with the MOT15 train split 
<br>
```eval.py``` <br>
```eval.py --policy path\to\trained\policy --datafolder path\tp\mot15\data --mode MARLMOT ``` <br>

A simulated version of SORT can be evaluated from comparison when mode is set to "SORT". Since MARLMOT is built upon SORT, the actions in this mode always default to a3 with probability of 1. (This is not exactly like SORT, but provides a nice easy comparison of the naive optimal policy). <br>
```eval.py  --mode SORT ```

##### NOTE: Evaluating in SORT mode is an effective way to quantify algorithm changes that are not directly related to MARLMOT

## Inference
To run inferences on either training or test data, inference.py can be called from the commandline which will create a video of all tracks
```inference.py --policy path\to\trained\policy --savepath path\to\save\video```
