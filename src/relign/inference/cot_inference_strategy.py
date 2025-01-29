import json

from relign.inference.tree_inference.branch_factor_strategy import (ListBranchFactor)
from relign.inference.tree_inference.expansion import NodeExpander
from relign.inference.tree_inference_strategy import TreeInferenceStrategy


class COTInferenceStrategy(TreeInferenceStrategy):
    def __init__(
        self,
        samples: int,
        node_expander: NodeExpander,
        **kwargs,
    ):
        super().__init__(
            node_expander=node_expander,
            **kwargs,
        )

        if self.cloud_logger is not None:
            self.cloud_logger.summary["branch_factor_strategy"] = json.dumps(
                self.node_expander.branch_factor_strategy.branch_factors
            )
