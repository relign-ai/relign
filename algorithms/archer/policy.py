import torch
from typing import Tuple
from policies.base_policy import BaseAgent
from policies.policy_factory import AgentFactory
from typing import List


class Archer(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_q(self, observation, action, detach_model=False):
        return self.critic.get_q(observation, action, detach_model=detach_model)

    def get_v(self, inputs, detach_model=False):
        return self.critic.get_v(inputs, detach_model=detach_model)

    def get_target_q(self, observation, action, detach_model=False):
        return self.target_critic.get_q(observation, action, detach_modesl=detach_model)

    def get_action(self, observation: List[str]):
        print("\n\getting action \n")
        print("observation: ", observation, "\n")
        print("\nobservations shape: ", len(observation))
        if self.template is not None:
            observation = [self.template.replace("{obs}, obs") for obs in observation]

        obs_ids = self.tokenizer(
            observation,
            return_tensors="pt",
            padding=True,
            max_length=512,
            truncation=True,
        ).to(self.device)

        context_len = obs_ids["attention_mask"].size(1)

        # Print the shape of obs_ids
        print("obs_ids shape:")
        for key, value in obs_ids.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")

        # Print the dimensions of the input_ids tensor
        print("input_ids dimensions:", obs_ids["input_ids"].dim())
        outputs = (
            self.accelerator.unwrap_model(self.model)
            .generate(
                **obs_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            .cpu()
        )

        outputs = outputs[:, :context_len:]
        raw_action = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        print("OUTPUT DIM: ", outputs)

        for _ in range(3):
            raw_action = [a[1:] if a.startswith("\n") else a for a in raw_action]
        if self.eos_str is not None:
            print("\n raw_action: ", raw_action, "\n")
            return [raw_a.split(self.eos_str)[0] for raw_a in raw_action]
        else:
            print("raw_action: ", raw_action)
            return raw_action

    def get_log_prob(self, observation, action):
        if self.template is not None:
            observation = [self.template.replace("{obs}", obs) for obs in observation]

        obs_ids = self.tokenizer(
            observation,
            return_tensors="pt",
            padding=True,
            max_length=512,
            truncation=True,
        ).to(self.device)

        action_ids = self.tokenizer(
            action, return_tensors="pt", padding=True, max_length=512, truncation=True
        ).to(self.device)

        input_ids = torch.cat([obs_ids["input_ids"], action_ids["input_ids"]], dim=1)

        attention_mask = torch.cat(
            [obs_ids["attention_mask"], action_ids["attention_mask"]], dim=1
        )
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        values = None
        if isinstance(outputs, Tuple):
            values, outputs = outputs
        prediction_probs = self.softmax(outputs.logits)

        selected_prediction_probs = torch.take_along_dim(
            prediction_probs[:, obs_ids["attention_mask"].size(1) - 1 : -1],
            action_ids["input_ids"].unsqueeze(2),
            dim=2,
        ).squeeze(2)

        if values is not None:
            return (
                values[:, obs_ids["attention_mask"].size(1) - 1 : -1],
                torch.log(selected_prediction_probs) * action_ids["attention_mask"],
                action_ids["attention_mask"],
            )
        else:
            return torch.sum(
                torch.log(selected_prediction_probs) * action_ids["attention_mask"],
                dim=1,
            )


AgentFactory.registerAgent(name="archer", agent=Archer)
