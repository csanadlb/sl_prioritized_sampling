import torch
from torch.utils.data import WeightedRandomSampler
from typing import Iterator
import copy


class CustomWeightedRandomSampler(WeightedRandomSampler):
    def __init__(self, c_const, explore_type, exploit_type, weights=None, num_samples=None, replacement=True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if weights is None:
            weights = torch.ones(num_samples, dtype=torch.double, device=device)
        super(CustomWeightedRandomSampler, self).__init__(
            weights,        # type: torch.tensor
            num_samples,    # type: int
            replacement     # type: bool
        )

        self.weights = weights
        self.fit_counts = torch.zeros(num_samples, dtype=torch.int32, device=device)
        self.init_pb_error = 0.001
        self.pb_errors = torch.ones(num_samples, dtype=torch.double, device=device) * self.init_pb_error
        self.indexes = torch.zeros(num_samples, dtype=torch.long, device=device)
        self.cp = c_const
        self.label_change_count = torch.zeros(num_samples, dtype=torch.double, device=device)
        self.labels = -1 * torch.ones(num_samples, dtype=torch.double, device=device)
        self.explore_type = explore_type
        self.exploit_type = exploit_type
        self.pb_log = {}

    def update_weights(self, indexes, outputs, predicted, labels, device):
        indexes = indexes.cpu()
        labels = labels.cpu()

        if self.explore_type == "fit_count":
            self.fit_counts[indexes] += 1

            log_term = torch.log(torch.max(self.fit_counts) + self.init_pb_error)
            explore = self.cp * torch.sqrt(2 * log_term / (self.fit_counts[indexes] + self.init_pb_error))

        else:
            raise NotImplementedError("Explore component not implemented")

        if self.exploit_type == "label_change":
            indexes_numpy = indexes.numpy()
            labels_dict = dict(zip(indexes_numpy, predicted.numpy()))

            for index in indexes_numpy:
                if labels_dict[index] != self.labels[index]:
                    self.label_change_count[index] = self.label_change_count[index] + 1
                    self.labels[index] = labels_dict[index]
                elif labels_dict[index] == self.labels[index] and self.label_change_count[index] == 0:
                    self.label_change_count[index] = self.label_change_count[index] + 1

            exploit = self.label_change_count[indexes_numpy] / torch.max(self.label_change_count)

        elif self.exploit_type == "pb_error":
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            # Create a mask for elements that are the same
            mask = (labels == predicted).float()
            pb_error = ((1 - mask) * outputs[torch.arange(outputs.size(0)), predicted] +
                        mask * (1 - outputs[torch.arange(outputs.size(0)), predicted])).to(device)

            self.pb_errors[indexes] = torch.abs(pb_error).double()
            exploit = self.pb_errors[indexes]
        else:
            raise NotImplementedError("Exploit component not implemented")

        self.weights[indexes] = (exploit + explore).detach()

        self.pb_log["exploit_mean"] = copy.copy(exploit.mean().item())
        self.pb_log["explore_mean"] = copy.copy(explore.mean().item())
        self.pb_log["exploit_std"] = copy.copy(exploit.std().item())
        self.pb_log["explore_std"] = copy.copy(explore.std().item())
        self.pb_log["exploit_max"] = copy.copy(exploit.max().item())
        self.pb_log["explore_max"] = copy.copy(explore.max().item())
        self.pb_log["exploit_min"] = copy.copy(exploit.min().item())
        self.pb_log["explore_min"] = copy.copy(explore.min().item())

        return self.pb_log

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        yield from iter(rand_tensor.tolist())

    def get_fit_count(self):
        return self.fit_counts
