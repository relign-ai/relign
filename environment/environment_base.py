from abc import ABC, abstractmethod
from typing import (
    Optional,
    List,
    Tuple,
    Generic,
    TypeVar,
    Dict,
    Union,
    NamedTuple,
    Self,
    Iterator,
    Callable,
    Any,
    Annotated,
)
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

# Basic RL Types
@dataclass(frozen=True)
class Reward(ABC):
    """Float scalar reward"""

    value: float

@dataclass(frozen=True)
class Done(ABC):
    """Bool scalar done (True if the episode is done)"""

    value: bool

# Modalities
@dataclass(frozen=True)
class Text:
    text: str
    is_action: bool

@dataclass(frozen=True)
class Image:
    image: np.ndarray
    is_action: bool

Modality = TypeVar("ModalityType")

@dataclass
class History(Generic[Modality]):
    history: Tuple[Modality, ...]

TextHistory = History[Text]
ImageHistory = History[Image]

text_history_to_str = lambda text_history: "".join(map(lambda x: x.text, text_history))
StepResult = Tuple[History[Modality], Reward, Done]

@dataclass(frozen=True)
class Trajectory(Generic[Modality]):
    history: History[Modality]
    reward: Reward
    done: Done

    def __post_init__(self):
        assert len(self.reward) == len(
            self.history
        ), "reward is needed for each trajectory entry"

        assert all(
            [r == 0.0 for r, t in zip(self.reward, self.history) if not t.is_action]
        ), "reward for non-actions texts should be 0.0"


# trajectory chain is a linked list of trajectories
@dataclass(frozen=True)
class TrajectoryChain(Generic[Modality]):
    trajectory: Trajectory[Modality]
    next: Optional[Self]

class Env(ABC, Generic[Modality]):
    @abstractmethod
    def step(self, history: History[Modality]) -> Tuple[History[Modality], Reward, Done]:
        pass

    @abstractmethod
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> History[Modality]:
        pass

    def close(self) -> None:
        pass

    def copy(self) -> Self:
        return deepcopy(self)


class BatchedEnv(ABC, Generic[Modality]):
    @abstractmethod
    def step(
        self,
        histories: List[Optional[History[Modality]]],
        done: Optional[List[Done]] = None,
    ) -> List[Optional[Tuple[History[Modality], Reward, Done]]]:
        pass

    @abstractmethod
    def reset(
        self,
        seed: Optional[List[Optional[int]]] = None,
        options: Optional[List[Optional[Dict]]] = None,
    ) -> List[History[Modality]]:
        pass

    def close(self) -> None:
        pass

    def copy(self) -> Self:
        return deepcopy(self)


# Adapter to convert Env to BatchedEnv
class EnvToBatchedEnv(BatchedEnv[Modality], Generic[Modality]):
    def __init__(self, env: Env[Modality], batch_size: int = 1):
        self.env = env
        self.batch_size = batch_size

    def step(
        self,
        histories: List[Optional[History[Modality]]],
        done: Optional[List[Done]] = None,
    ) -> List[Optional[Tuple[History[Modality], Reward, Done]]]:
        results = []
        for history, done_flag in zip(histories, done or [False]*self.batch_size):
            if done_flag or history is None:
                results.append(None)
            else:
                step_result = self.env.step(history)
                results.append(step_result)
        return results

    def reset(
        self,
        seed: Optional[List[Optional[int]]] = None,
        options: Optional[List[Optional[Dict]]] = None,
    ) -> List[History[Modality]]:
        reset_histories = []
        for _ in range(self.batch_size):
            history = self.env.reset()
            reset_histories.append(history)
        return reset_histories
    

# Policy and BatchedPolicy remain generic or can also be parameterized similarly
class Policy(ABC, Generic[Modality]):
    """
    Policy is a policy that takes a single history.
    """
    @abstractmethod
    def act(self, history: History[Modality]) -> History[Modality]:
        pass



class BatchedPolicy(ABC, Generic[Modality]):
    """
    Batched policy is a policy that takes a batch size list of histories. 
    """
    @abstractmethod
    def act(self, history: List[Optional[History[Modality]]]) -> List[Optional[History[Modality]]]:
        pass


class PolicyToBatchedPolicy(BatchedPolicy[Modality], Generic[Modality]):
    """
    Policy to batched policy is a batched policy that takes a single policy.
    """
    def __init__(self, policy: Policy[Modality]):
        self.policy = policy
    
    def act(self, history: List[Optional[History[Modality]]], done: Optional[List[Done]]=None) -> List[Optional[History[Modality]]]:
        print(done)
        print(history)
        print(len(history))
        if done is None:
            done = [False]*len(history)
        assert len(history) == len(done)
        return [None if d else self.policy.act(item) for item, d in zip(history, done)]


class BatchedPolicyToPolicy(Policy[Modality], Generic[Modality]):
    """
    Given a batched policy we perofm the policy on the batch and return 
    a single . 
    """
    def __init__(self, policy: BatchedPolicy[Modality]):
        self.policy = policy
    
    def act(self, history: History[Modality]) -> History[Modality]:
        return self.policy.act([history])[0]


# interact
class InteractionTransition(NamedTuple):
    pre_action_history: Annotated[History[Modality], "history before action"]
    post_action_history: Annotated[History[Modality], "history after action"]
    post_transition_history: Annotated[History[Modality], "history after environment step"]
    reward: Annotated[Reward, "reward given from the environment step"]
    done: Annotated[Done, "done signal from the environment step"]


def interact_environment(
    env: Union[Env[Modality], BatchedEnv[Modality]],
    policy: Union[Policy[Modality], BatchedPolicy[Modality]],
    initial_history: Optional[Union[History[Modality], List[History[Modality]]]] = None,
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
    if isinstance(policy, Policy):
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
        history = policy.act(padded_history)
        history = history[:bsize]
        post_action_history = history

        # Step the environment
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

def env_eval(
    env: Union[Env[Modality], BatchedEnv[Modality]], 
    policy: Union[Policy[Modality], BatchedPolicy[Modality]], 
    n_rollouts: int, 
    initial_text_history: Optional[History[Modality]] = None, 
    seed_generator: Optional[Iterator[int]] = None, 
    env_options: Optional[Dict] = None, 
    interaction_callback: Optional[Callable[[List[InteractionTransition]], None]] = None, 
    bsize: int = 1, 
    verbose: bool = True, 
) -> Tuple[List[List[InteractionTransition]], Dict[str, Any]]:
    interactions, rewards, dones, eps_lengths = [], [], [], []
    for _ in tqdm(range((n_rollouts + (bsize - 1)) // bsize), disable=not verbose):
        actual_bsize = min(n_rollouts - len(interactions), bsize)
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