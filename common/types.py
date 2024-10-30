from abc import ABC
from typing import (
    Tuple,
    Generic,
    TypeVar,
    Optional,
    List,
    Annotated,
    NamedTuple
)
from dataclasses import dataclass
import numpy as np
from transformers import PreTrainedTokenizer

# Basic RL Lingo
@dataclass(frozen=True)
class Reward(ABC):
    """Float scalar reward"""
    value: float

@dataclass(frozen=True)
class Done(ABC):
    """Bool scalar done (True if the episode is done)"""
    value: bool

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
    """
        A history is a sequence of interweaved states and actions.  
        The history is used to represent the state of the environment. 
    """
    history: Tuple[Modality, ...]
    

TextHistory = History[Text]
ImageHistory = History[Image]

text_history_to_str = lambda text_history: "".join(map(lambda x: x.text, text_history))
StepResult = Tuple[History[Modality], Reward, Done]


@dataclass(frozen=True)
class Trajectory():
    """
        A trajectory is a sequence of observations actions and rewards. 
        The history is an interweaved sequence of states from the environment and actions taken by the policy. 
        The rewards for non-actions are always 0.0
    """
    history: History[Modality]
    reward: Tuple[Reward, ...]
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
class TrajectoryChain():
    trajectory: Trajectory
    next: Optional['TrajectoryChain']


@dataclass(frozen=True)
class TokenTrajectory():
    tokens: np.ndarray # 1d int 32 array
    is_action: np.ndarray # 1d bool array
    reward: np.ndarray # 1d float
    done: np.ndarray # bool scalar

    def __post_init__(self):
        assert len(self.tokens.shape) == 1, 'tokens must be 1 dimensional'
        assert len(self.is_action.shape) == 1, 'is_action must be 1 dimensional'
        assert len(self.reward.shape) == 1, 'reward must be 1 dimensional'
        assert len(self.done.shape) == 0, 'done must be scalar'

        assert self.is_action.shape == self.tokens.shape, 'is_action must have the same shape as tokens'
        assert self.reward.shape == self.tokens.shape, 'reward must have the same shape as tokens'

        assert not np.any(((1 - self.is_action.astype(np.float32)) * self.reward) != 0.0), 'reward must be 0.0 if not an action'

    @classmethod
    def from_text_trajectory(cls, trajectory: Trajectory, tokenizer: PreTrainedTokenizer) -> 'TokenTrajectory':
        """
            Instantiate a token trajectory from a text trajectory
        """

        tokens = []
        is_action = []
        rewards = []

        for i, item in enumerate(trajectory.history):
            new_tokens = tokenizer.encode(item.text)
            tokens.extend(new_tokens) 
            is_action.extend([item.is_action]* len(new_tokens)) 
            rewards.extend([0.0] * len(new_tokens-1) + [trajectory.reward[i]])
        
        done = trajectory.done.value

        tokens = np.array(tokens, dtype=np.int32)
        actions = np.array(is_action, dtype=np.bool_)
        rewards = np.array(rewards, dtype=np.float32)
        done = np.array(done, np.bool_)

        return cls(tokens, actions, rewards, done)

class TextEpisode:
    query: TextHistory  # History of text observations and actions
    response: Text # Text observation 
    score: float = None # score after the action
    advantage: float = None # advantagges 

@dataclass  
class Episode:
    query_token_ids: List[int]
    response_token_ids: List[int]
    reward: float = None
    advantages: Optional[List[float]] = None

    def __post_init__(self):
        if self.advantages is not None:
            assert len(self.advantages) == len(self.response_token_ids)

# interact
class InteractionTransition(NamedTuple):
    pre_action_history: Annotated[History, "history before action"]
    post_action_history: Annotated[History, "history after action"]
    post_transition_history: Annotated[History, "history after environment step"]
    reward: Annotated[Reward, "reward given from the environment step"]
    done: Annotated[Done, "done signal from the environment step"]





