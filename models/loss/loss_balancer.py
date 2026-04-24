import torch


class LossBalancer:
    def __init__(self, beta=0.9, clamp=(0.1, 5.0)):
        self.beta = beta          # EMA decay
        self.clamp = clamp
        self.ema_eta = None
        self.ema_cl = None

    def __call__(self, loss_eta, loss_cl, task_beta):
        with torch.no_grad():
            val_eta = loss_eta.item()
            val_cl = loss_cl.item()

            if self.ema_eta is None:
                self.ema_eta = val_eta
                self.ema_cl = val_cl
            else:
                self.ema_eta = self.beta * self.ema_eta + (1 - self.beta) * val_eta
                self.ema_cl  = self.beta * self.ema_cl  + (1 - self.beta) * val_cl

            scale = (self.ema_eta / (self.ema_cl + 1e-6))
            scale = max(self.clamp[0], min(self.clamp[1], scale))

        return task_beta * loss_eta + (1 - task_beta) * loss_cl * scale