"""
    Custom PPO implementation

    TODO: implement early stopping upon convergence
"""

import os
import csv
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

# import track utils
from track_utils import *



class PPO():
    def __init__(self, dataloader, env, policy_model, epochs, num_train_iters=4, 
                 lr=1e-4, gamma=0.95, eps=0.2, iou_threshold=0.3, min_age=3, 
                 device=None, checkpoint=50, obs_dim=18, action_dim=5):
        """
            Custom Class for Proximal Policy Optimization for MARLMOT
            Assumes a continuous observation space and a discrete action space.
            Args:
                dataloader - Custom DataLoader with training data
                env - custom environment for MARLMOT
                actor - Actor model for the policy
                critic - Critic model for the value function
                policy_model - NN model class for actor and critic
                epochs - number of epochs for outer training loop
                num_train_iters - number of iterations to optimize actor and critic
                lr - learning rate for model optimizers
                gamma - discount rate for computing rewards-to-go
                eps - clip factor for clipped PPO loss function
                min_age - miniumum age for a track to be considered valid
                iou_threshold - min IOU threshold for track association
                device - device to perform inference on (defaults to GPU if available)
                checkpoint - number of epochs to sa
                obs_dim - observation dimensions
                action_dim - action dimensions
        """
        self.dataloader = dataloader
        self.env = env
        self.epochs = epochs
        self.num_train_iters = num_train_iters
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.iou_threshold = iou_threshold
        self.min_age = min_age
        self.checkpoint = checkpoint if checkpoint else epochs

        # store training metrics
        self.metrics = {
               "action_ratios" : [], # ratios of each action per batch
             "false_positives" : [], # total number of false positives per batch
             "false_negatives" : [], # total number of false negatives (missed tracks) per batch
            "mismatch_errrors" : [], # total number of mismatch errors per batch
                "cost_penalty" : [], # total cost of all matches per batch (smaller is better)
                        "mota" : [], # multiple object tracking accuracy
                   "approx_kl" : []  # approximate KL divergence
        }

        # get device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ## STEP 1
        # initialize actor and critic
        self.actor = policy_model(input_dim=obs_dim, output_dim=action_dim).to(self.device)
        self.critic = policy_model(input_dim=obs_dim, output_dim=1).to(self.device)
        
        # initialize optimizers for actor and critic
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        # set up logger for storing training info
        self.logger = {
                  "delta_t" : [], # time stamp at the end of each epoch
            "batch_lengths" : [], # length of collected batch data
               "avg_return" : [], # average undiscoutned return per epoch
             "actor_losses" : [], # actor losses
            "critic_losses" : []  # actor losses
        }
        

    def train(self, savepath=None):
        """
            Main (outer) training loop for PPO

            Collects a batch of trajectories/rollouts (each rollout is a single video)
            Computes advantages and normalizes them for stability
            Updates the actor and critic weights

            Inputs: savepath - (str) denotes where to save model after each epoch 
                        models will not be saved if not provided

            To load trained policy:
                actor = policy_model(input_dim=obs_dim, output_dim=action_dim).to(self.device)
                actor.load_state_dict(torch.load(savepath))
                actor.eval();
        """
        for epoch in range(self.epochs):
            ## STEP 3 & 4
            # compute batch rollouts/episodes/trajectories 
            batch_obs, batch_acts, batch_log_probs, batch_rtgs = self.batch_rollout()
            
            ## STEP 5 compute advantages
            V, _ = self.compute_value(batch_obs, batch_acts)

            # detach from gradient important for computing actor loss
            A_k = batch_rtgs - V.detach() 

            # standardize advantages
            # A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            ## Optimize actor and critic for n steps
            # can't do too many steps or assumptions of similar distributions will fail
            # can probably compute KL divergence after this loop to help with debugging if needed
            for _ in range(self.num_train_iters):

                ## STEP 6 & 7 Optimize Actor and Critic

                # compute V and new action logprobs with updated policy
                V, curr_log_probs = self.compute_value(batch_obs, batch_acts)
                
                # compute importance sampling ratio of probs
                # new policy / old policy --> it is important for these to be similar :)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # compute actor loss 
                surrogate_1 = ratios * A_k 
                surrogate_2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * A_k
                actor_loss = -torch.mean(torch.min(surrogate_1, surrogate_2))
                
                # compute critic loss
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # backpropagate losses and step weights for actor and critic
                # set retain_graph to True to step critic weights after actor weights
                # see: https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            # compute KL divergence
            self.metrics["approx_kl"].append(
                (curr_log_probs - batch_log_probs).mean().cpu().item())


            # update logger 
            self.logger["delta_t"].append(time.time())
            self.logger["actor_losses"].append(
                actor_loss.detach().cpu().numpy().tolist())
            self.logger["critic_losses"].append(
                critic_loss.detach().cpu().numpy().tolist())

            # display epoch results
            print("epoch: ", epoch)
            print("Number of Batch samples: ", 
                  self.logger["batch_lengths"][-1])
            print("Actor Loss: {:.5f}".format(self.logger["actor_losses"][-1]))
            print("Critic Loss: {:.5f}".format(self.logger["critic_losses"][-1]))
            print("Avg Return: {:.5f}".format(self.logger["avg_return"][-1]))
            print()

            # save policy
            if savepath:
                torch.save(self.actor.state_dict(), 
                           os.path.join(savepath, f"actor_{epoch}.pth"))

            if savepath and ((epoch + 1) % self.checkpoint == 0):
                self.save_logger(savepath)
                self.save_hyperparameters(savepath)
                self.save_metrics(savepath)

        # # save final 
        # self.save_logger(savepath)
        # self.save_hyperparameters(savepath)
        # self.save_metrics(savepath)


    def batch_rollout(self):
        """ 
            Inner training loop:
                collects all rollouts for a single batch. Just like the MARLMOT
		        paper, we will collect rollouts from all training videos for each batch

            For each video, we will collect the S, A, R tuple for each frame.
            NOTE: for the final frame we could take an action, but we don't have any truth 
            data to quantify the rewards, so we leave it out. The length for each video
            should be the total number of frames minus 1.
            
            Inputs: None
            Outputs:
                batch_obs - Nx18 tensor of batch observations
                batch_actions - Nx1 tensor of batch actions
                batch_logprobs - Nx1 tensor of batch action probabilities
                batch_rtgs - Nx1 tensor of computed Batch Rewards-to-Go 

                Note: The length of 'N' will change on each iteration based on the actions
                    taken by the agent. 
        """
        batch_obs = []
        batch_actions = []
        batch_logprobs = []
        batch_rewards = []
        batch_rtgs = []

        # store metrics
        num_false_positives = 0
        num_false_negatives = 0
        num_mismatch_errrors = 0
        cost_penalties = 0
        total_num_tracks = 0 # total number of gt tracks for all frames

        for (ground_truth, detections, frame_size) in self.dataloader:
            
            # initialize world object to collect rollouts
            tracker = HungarianTracker(iou_threshold=self.iou_threshold, 
                                       min_age=self.min_age)
            world = self.env(tracker=tracker, 
                             ground_truth=ground_truth, 
                             detections=detections,
                             frame_size=frame_size)

            # initialize episode rewards list
            ep_rewards = []

            # accumulate total number of tracks for mota
            total_num_tracks += len(ground_truth)

            # take initial step to get first observations
            observations, _, _ = world.step({})

            # collect (S, A, R) trajectory for entire video
            while True:    

                # append observations first
                batch_obs += list(observations.values())

                # take actions
                actions, logprobs = self.get_actions(observations)
                # get rewards and new observations
                observations, rewards, done = world.step(actions)

                # get metrics
                num_false_positives += len(world.false_positives)
                num_false_negatives += len(world.missed_tracks)
                num_mismatch_errrors += world.mismatch_errors
                cost_penalties += world.cost_penalty

                # store actions and new rewards 
                batch_rewards.append(rewards)
                batch_actions += list(actions.values())
                batch_logprobs += logprobs

                # assume that tracks at each frame occur at the same time step
                ep_rewards.append(rewards) 

                if done:
                    break
                
            ## STEP 4 compute rewards to go for trajectory
            batch_rtgs += self.compute_frame_rtgs(ep_rewards) 

        # compute and store metrics
        action_ratios = np.zeros((5,))
        unique_actions, action_counts = np.unique(batch_actions, 
                                                  return_counts=True)
        action_ratios[unique_actions] = action_counts/len(batch_actions)

        mota = 1 - ((num_false_positives 
                     + num_false_negatives 
                     + num_mismatch_errrors)) / total_num_tracks

        self.metrics["action_ratios"].append(action_ratios)
        self.metrics["false_positives"].append(num_false_positives)
        self.metrics["false_negatives"].append(num_false_negatives)
        self.metrics["mismatch_errrors"].append(num_mismatch_errrors)
        self.metrics["cost_penalty"].append(cost_penalties)
        self.metrics["mota"].append(mota)

        # convert everything to float32 torch tensors and place them on the device
        batch_obs = torch.tensor(np.array(batch_obs).squeeze(), 
                                 dtype=torch.float32).to(self.device)
        batch_actions = torch.tensor(np.array(batch_actions), 
                                     dtype=torch.float32).to(self.device)
        batch_logprobs = torch.tensor(batch_logprobs, 
                                      dtype=torch.float32).to(self.device)
        batch_rtgs = torch.tensor(np.array(batch_rtgs), 
                                  dtype=torch.float32).to(self.device)

        # update logger
        self.logger["batch_lengths"].append(len(batch_actions))
        self.logger["avg_return"].append(
            np.mean([sum(rew) for rew in batch_rewards if len(rew) > 0]))
                        
        return batch_obs, batch_actions, batch_logprobs, batch_rtgs
            

    def get_actions(self, observations):
        """
            Obtains actions and logprobs from current observations
            Inputs:
                observations - (dict) maps track ids to (18x1) observation vectors
            Outputs:
                actions - (dict) maps track ids to discrete actions for observations
                logprobs -- (tesnor) log probabilities of each action
        """
        # handle initial frame where no observations are made
        if len(observations) == 0:
            return {}, []
        
        # apply policy on current observations 
        obs = torch.tensor(np.array(list(observations.values())).squeeze(), 
                           dtype=torch.float).to(self.device)
        logits = self.actor(obs)

        # get actions
        dist = Categorical(logits=logits)
        actions = dist.sample()

        # get logprob of each action
        logprobs = dist.log_prob(actions) 

        # map track IDs to actions
        try:
            actions = dict(zip(observations.keys(), 
                               actions.cpu().numpy()))
        except TypeError:
            # handle case for length 1 observation
            actions = dict(zip(observations.keys(), 
                               [actions.cpu().numpy().tolist()]))
            logprobs = logprobs.unsqueeze(0)
        
        return actions, logprobs
    

    def compute_frame_rtgs(self, rewards):
        """
            Computed Rewards-to-Go from an episode of frame rewards. 
            
            The frame reward list contains N sublists that each correspond to a 
            single frame. The number of rewards at each frame depends on 
            the current number of tracks and can vary in size. However, the
            rewards-to-go will be computed as if each sublist was a single 
            reward. i.e. if gamma=0.9
                rewards = [[-2, -2, -2], [-3, -3], [-1, -1, -1]]  
                rtgs = [[-5.51, -5.51, -5.51], [-3.9, -3.9], [-1.0, -1.0, -1.0]]

            Inputs:
                rewards - a list of length N containing sublists of rewards
                    for each frame, sublists can have variable length
            Outputs: 
                rtgs - A 1D computed rewards to go list with a flattened shape of
                    the input rewards list
        """
        rtgs = []
        discounted_reward = 0 # The discounted reward so far

        # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
        # discounted return (think about why it would be harder starting from the beginning)
        for rew in reversed(rewards):
            if len(rew) == 0:
                # print("0 rewards!")
                continue

            # this is faster than numpy for small lists
            discounted_reward = [r + discounted_reward*self.gamma for r in rew]
            rtgs += discounted_reward
            discounted_reward = discounted_reward[0]
        
        return list(reversed(rtgs))
    

    def compute_rtgs(self, rewards):
        """
            Computed Rewards-to-Go from an episode of frame. 
            
            The frame reward list contains N sublists that each correspond to a 
            single frame. The number of rewards at each frame depends on 
            the current number of tracks and can vary in size. However, the
            rewards-to-go will be computed as if each sublist was a single 
            reward. i.e. if gamma=0.9
                rewards = [-2, -3, -1] 
                rtgs = [-5.51, -3.9 -1.0]

            Inputs:
                rewards - A length N  list of rewards
            Outputs: 
                rtgs - A length N list of computed rewards-to-go
        """
        rtgs = []
        discounted_reward = 0 # The discounted reward so far

        # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
        # discounted return (think about why it would be harder starting from the beginning)
        for rew in reversed(rewards):

            discounted_reward = rew + discounted_reward*self.gamma
            rtgs.insert(0, discounted_reward)
        
        return rtgs
    

    def compute_value(self, batch_obs, batch_acts):
        """ 
            Compute Value estimatation from current observations and actions
            Inputs:
                batch_obs - batch observations 
                batch_acts - batch actions
            Outputs:
                V - estimated value function for the current batch
                logprobs - newly computed log probabilities for the batch actions
                    This is important since our actor is constantly getting weight
                    updates.
        """
        # compute values for batch observations
        V = self.critic(batch_obs).squeeze()

        # get a sample of logprobs from the batch actions
        logits = self.actor(batch_obs)
        dist = Categorical(logits=logits)
        logprobs = dist.log_prob(batch_acts)

        return V, logprobs
    

    def save_hyperparameters(self, savepath):
        """ Creates a Dictionary of hyperparameters and saves it to save path """
        hyperparameters = {
            "epochs" : self.epochs,
            "num_train_iters" : self.num_train_iters,
            "learning_rate" : self.lr,
            "discount_factor" : self.gamma,
            "clip factor" : self.eps,
            "iou_threshold" : self.iou_threshold,
            "min_age" : self.min_age,
            "device" : self.device,
        }
        with open(os.path.join(savepath, "hyperparameters.csv"), 'w') as fout:
            writer = csv.DictWriter(fout, fieldnames=hyperparameters.keys())
            writer.writeheader()
            writer.writerow(hyperparameters)
    

    def save_logger(self, savepath):
        """ Saves the logger dictionary to savepath """
        df = pd.DataFrame(self.logger)
        df.to_csv(os.path.join(savepath, "logger.csv"))


    def save_metrics(self, savepath):
        """ Saves metrics to the save path """
        action_ratios = np.vstack(self.metrics['action_ratios'])
        action_0_ratio = action_ratios[:, 0]
        action_1_ratio = action_ratios[:, 1]
        action_2_ratio = action_ratios[:, 2]
        action_3_ratio = action_ratios[:, 3]
        action_4_ratio = action_ratios[:, 4]

        cols = ['action_0_ratio', 
                'action_1_ratio', 
                'action_2_ratio', 
                'action_3_ratio', 
                'action_4_ratio', 
                'false_positives', 
                'false_negatives', 
                'mismatch_errrors', 
                'cost_penalty', 
                'approx_kl']

        arr = np.vstack([action_0_ratio,
                        action_1_ratio,
                        action_2_ratio,
                        action_3_ratio,
                        action_4_ratio,
                        np.array(self.metrics['false_positives']),
                        np.array(self.metrics['false_negatives']),
                        np.array(self.metrics['mismatch_errrors']),
                        np.hstack(self.metrics['cost_penalty']),
                        np.array(self.metrics['approx_kl'])]).T

        df = pd.DataFrame(arr, columns=cols)
        df.to_csv(os.path.join(savepath, "metrics.csv"))


    def _init_hyperparameters(self, hyperparameters):
        """ initializes hyperparameters for PPO """

        self.epochs = hyperparameters["epochs"]
        self.num_train_iters = hyperparameters["num_train_iters"]
        self.lr = hyperparameters["lr"]
        self.gamma = hyperparameters["gamma"]
        self.eps = hyperparameters["eps"]
        self.iou_threshold = hyperparameters["iou_threshold"]
        self.min_age = hyperparameters["min_age"]

        