class PathFilter:
    def __call__(self, path) -> bool:
        raise NotImplementedError


class NonePathFilter(PathFilter):
    def __call__(self, path):
        return True


class SuccessfulPathFilter(PathFilter):
    def __call__(self, path):
        last_node = path["node_chain"][-1]
        if "answer" not in last_node:
            return False
        return last_node["is_correct_answer"]


class NonZeroScorePathFilter(PathFilter):
    def __call__(self, path):
        first_step_node = path["node_chain"][1]
        return first_step_node["score"] > 0


class NonZeroLastStepScorePathFilter(PathFilter):
    def __call__(self, path):
        last_node = path["node_chain"][-1]
        return last_node["score"] > 0


class ZeroLastStepScorePathFilter(PathFilter):
    def __call__(self, path):
        last_node = path["node_chain"][-1]
        return last_node["score"] == 0


class NonZeroLastStepAdvantagePathFilter(PathFilter):
    def __call__(self, path):
        last_node = path["node_chain"][-1]
        return last_node["advantage"] > 0
