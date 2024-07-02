import torch


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2):  # noqa
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean(torch.pow(euclidean_distance, 2))
        return loss
