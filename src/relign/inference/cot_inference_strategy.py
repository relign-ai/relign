import json

from relign.inference.tree_inference.branch_factor_strategy import ListBranchFactor
from relign.inference.tree_inference.expansion import NodeExpander
from relign.inference.tree_inference_strategy import TreeInferenceStrategy


from relign.utils.logging import get_logger

logger = get_logger(__name__)


class COTInferenceStrategy(TreeInferenceStrategy):
    def __init__(
        self,
        samples: int,
        node_expander: NodeExpander,
        **kwargs,
    ):
        """Chain of thought inference is a listBranchfactor witn a single root/question node
        branching out to `samples` single path reasoning traces.

        Therefore we overwrite the branchfactor strategy
        """
        logger.warning(
            "The old branch_factor_strategy is overwritten by COTInferenceStrategy"
        )

        branch_factor_strategy = ListBranchFactor(
            branch_factors=[
                {
                    "depth": 0,
                    "branch_factor": samples,
                },  # Branch out at the root node (i.e., the question)
                {"depth": 1, "branch_factor": 1},  # follow a single path
            ],
        )
        node_expander.branch_factor_strategy = branch_factor_strategy

        super().__init__(
            node_expander=node_expander,
            **kwargs,
        )

        if self.cloud_logger is not None:
            self.cloud_logger.summary["branch_factor_strategy"] = json.dumps(
                self.node_expander.branch_factor_strategy.branch_factors
            )
