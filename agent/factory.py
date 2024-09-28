from typing import Type, Dict

from agent.base import AgentBase
import torch
import accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.print import colorful_print


class AgentFactory:
    _agents: Dict[str, Type[AgentBase]] = {}

    @classmethod
    def registerAgent(self, agent: Type[AgentBase]):
        """
        Register an agent class with the factory.

        Args:
            agent (Type[Agent]): The agent class to register.
        """
        colorful_print(f"Registering agent {agent.__name__}", fg="green")
        self._agents[agent.__name__] = agent

    @classmethod
    def create_agent(cls, name: str, config, *args, **kwargs) -> AgentBase:
        """
        Create an instance of the agent with the given name.

        Args:
            name (str): The name of the agent.
            config: Configuration object containing agent parameters.

        Returns:
            Agent: An instance of the requested agent.
        """
        if name not in cls._agents:
            raise ValueError(f"Agent '{name}' not registered.")
        agent_class = cls._agents[name]

        # Extract agent configuration
        policy_lm = config.agent.get("policy_lm", "gpt2")
        cache_dir = config.agent.get("cache_dir", "~/.cache")
        use_lora = config.agent.get("use_lora", False)
        use_bfloat16 = config.agent.get("use_bfloat16", False)

        # Initialize device and accelerator
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        accelerator = accelerate.Accelerator()

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(policy_lm, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            policy_lm,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16 if use_bfloat16 else None,
        ).to(device)

        # Create agent instance
        agent = agent_class(
            model=model,
            tokenizer=tokenizer,
            critic=None,  # Initialize critic as needed
            double_critic=False,  # Update based on your needs
            device=device,
            accelerator=accelerator,
            policy_lm=policy_lm,
            cache_dir=cache_dir,
            use_lora=use_lora,
            use_bfloat16=use_bfloat16,
            # Add other parameters from config as needed
        )
        return agent
