from abc import ABC, abstractmethod


class BasePruner(ABC):
    def __init__(self):
        self.layers = []

    def register(self, layer):
        self.layers.append(layer)

    @abstractmethod
    def prune(self, *args, **kwargs):
        ...


class Pruner(BasePruner):
    def prune(self, prob: float):
        print("Pruning step")
        for layer in self.layers:
            layer.prune(prob)

    def decrement_immunity(self):
        for layer in self.layers:
            if hasattr(layer, "decrement_immunity"):
                layer.decrement_immunity()


class RetrainPruner:
    def __init__(self):
        self.layers = []

    def register(self, layer):
        self.layers.append(layer)

    def pre_retrain(self):
        print("Pre retrain")
        for layer in self.layers:
            layer.pre_retrain()

    def post_retrain(self):
        print("Post retrain")
        for layer in self.layers:
            layer.post_retrain()

    def prepare_new(self, prob: float):
        print("Preparing new step")
        for layer in self.layers:
            layer.prepare_new_weights(prob)

    def apply_new_weights(self):
        print("Applying new weights")
        for layer in self.layers:
            layer.apply_new_weights()
