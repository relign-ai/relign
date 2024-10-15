from typing import Callable, Optional, List, Dict, Union, Tuple, Any, Iterator

from tqdm import tqdm
import numpy as np

from policies.base_policy import BasePolicy, BatchedPolicy
from episode_generators.environment.base_environment import Env, BatchedEnv, InteractionTransition, EnvToBatchedEnv
from common.types import Trajectory, History


def add_trajectory_reward(trajectory):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_reward = np.sum([d["reward"] for d in trajectory])
    for d in trajectory:
        d.update({"trajectory_reward": trajectory_reward})
    return trajectory


def add_mc_return(trajectory, gamma=0.95):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_rewards = np.array([d["reward"] for d in trajectory]).reshape(1, -1)
    gamma_row = np.cumprod(np.ones((1, trajectory_rewards.shape[1])) * gamma)
    gamma_matrix = np.triu(gamma_row.reshape(1, -1) / gamma_row.reshape(-1, 1))
    mc_returns = np.sum(trajectory_rewards * gamma_matrix, axis=1)
    for d, mc in zip(trajectory, mc_returns):
        d.update({"mc_return": mc})
    return trajectory


def interact_environment(
    env: Union[Env, BatchedEnv],
    policy: Union[BasePolicy, BatchedPolicy],
    initial_history: Optional[Union[History,List[History]]] = None,
    env_seed: Union[Optional[int], Optional[List[Optional[int]]]] = None,
    env_options: Union[Optional[Dict], Optional[List[Optional[Dict]]]] = None,
    bsize: int = 1,
    npad: int = 0,
) -> List[List[InteractionTransition]]:
    """
    Interact with the environment using the policy.
    """
    assert bsize > 0, "batch size must be greater than 0"

    # Convert to batched environment if necessary
    if isinstance(env, Env):
        env = EnvToBatchedEnv(env, batch_size=bsize)
    elif isinstance(env, BatchedEnv):
        assert env.batch_size == bsize, "Batch size mismatch."
    # Convert to batched policy if necessary
    if isinstance(policy, BasePolicy):
        policy = PolicyToBatchedPolicy(policy, batch_size=bsize)
    elif isinstance(policy, BatchedPolicy):
        assert policy.batch_size == bsize, "Batch size mismatch."

    # Handle seeds and options
    if env_seed is not None and isinstance(env_seed, int):
        env_seed = [env_seed] * bsize
    if env_options is not None and isinstance(env_options, dict):
        env_options = [env_options] * bsize
    if initial_history is not None and not isinstance(initial_history, list):
        initial_history = [initial_history] * bsize

    history = initial_history
    if history is None:
        history = env.reset(env_seed, env_options)

    transitions_batch = [[] for _ in range(bsize)]
    done = [False] * bsize

    while not all(done):
        pre_action_history = history
        # Pad histories if necessary
        padded_history = history + [(Text("", is_action=False),)] * npad
        
        # Get actions from the policy
        history = policy.predict(padded_history)
        history = history[:bsize]
        post_action_history = history

        # Get result from environment
        step_results = env.step(history, done=done)
        step_results = [
            (None, None, True) if x is None else x for x in step_results
        ]
        history, reward, done = zip(*step_results)
        history = list(history)
        reward = list(reward)
        done = list(done)
        post_transition_history = history

        for batch_idx in range(bsize):
            if done[batch_idx] and (
                pre_action_history[batch_idx] is None
                or post_action_history[batch_idx] is None
                or post_transition_history[batch_idx] is None
                or reward[batch_idx] is None
            ):
                continue
            transitions_batch[batch_idx].append(
                InteractionTransition(
                    pre_action_history=pre_action_history[batch_idx],
                    post_action_history=post_action_history[batch_idx],
                    post_transition_history=post_transition_history[batch_idx],
                    reward=reward[batch_idx],
                    done=done[batch_idx],
                )
            )
    return transitions_batch


def collect_rollouts(
    policy: Union[BasePolicy, BatchedPolicy], 
    env: Union[Env, BatchedEnv], 
    num_rollouts: int, 
    initial_text_history: Optional[History] = None, 
    seed_generator: Optional[Iterator[int]] = None, 
    env_options: Optional[Dict] = None, 
    interaction_callback: Optional[Callable[[List[InteractionTransition]], None]] = None, 
    bsize: int = 1, 
    verbose: bool = True, 
) -> Tuple[List[List[InteractionTransition]], Dict[str, Any]]:
    """
    Make an agent interact with an envonment in a batched fashion. Returns a list of batch size 
    interactions. 
    """
    interactions, rewards, dones, eps_lengths = [], [], [], []
    for _ in tqdm(range((num_rollouts + (bsize - 1)) // bsize), disable=not verbose):
        actual_bsize = min(num_rollouts - len(interactions), bsize)
        npad = bsize - actual_bsize
        interaction_batch = interact_environment(
            env, 
            policy, 
            initial_history=initial_text_history, 
            env_seed=[None] * actual_bsize if seed_generator is None else [next(seed_generator) for _ in range(actual_bsize)], 
            env_options=[env_options] * actual_bsize, 
            bsize=actual_bsize,
            npad=npad,
        )
        
        for interaction in interaction_batch:
            interactions.append(interaction)
            rewards.append(sum(x.reward.value for x in interaction))
            dones.append(interaction[-1].done.value)
            eps_lengths.append(len(interaction))
            if interaction_callback is not None:
                interaction_callback(interaction)
    
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    results_summary = dict(
        reward=dict(
            mean=np.mean(rewards), 
            std=np.std(rewards), 
            min=np.min(rewards), 
            max=np.max(rewards), 
        ), 
        done=dict(
            mean=np.mean(dones), 
            std=np.std(dones), 
            min=np.min(dones), 
            max=np.max(dones), 
        ), 
        length=dict(
            mean=np.mean(eps_lengths),
            std=np.std(eps_lengths),
            min=np.min(eps_lengths),
            max=np.max(eps_lengths),
        ),
    )
    return interactions, results_summary


def get_trajectory_from_transitions(transitions: List[InteractionTransition]) -> Trajectory:
    """
        Convert a list of InteractionTransitions into a Trajectory.
    """
    rewards = []
    for transition in transitions:
        rewards.append(transition.reward)
    history = transitions[-1].post_transition_history
    done = transitions[-1].done
    return Trajectory(history=history, rewards=rewards, done=done)


def get_batch_trajectories(transitions: List[List[InteractionTransition]]) -> List[Trajectory]:
    """
        Convert a list of lists of InteractionTransitions into a list of Trajectories.
    """
    trajectories = []
    for transition in transitions:
        trajectories.append(get_trajectory_from_transitions(transition))
    return trajectories


