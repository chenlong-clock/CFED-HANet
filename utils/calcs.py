import torch
from sklearn.metrics import f1_score
class Calculator():
    def __init__(self):
        self.predict = torch.Tensor()
        self.label = torch.Tensor()
        # TODO:test use
        # self.predict = torch.Tensor().to('cpu')
        # self.label = torch.Tensor().to('cpu')
    def extend(self, predict, label):
        # TODO:test use
        self.predict = torch.cat([self.predict, predict.to('cpu')])
        self.label = torch.cat([self.label, label.to('cpu')])

    def micro_F1_Score(self):
        return f1_score(self.label.detach().numpy(), self.predict.detach().numpy(), average='micro')
        
    def marco_F1_Score(self):
        return f1_score(self.label.detach().numpy(), self.predict.detach().numpy(),average='macro')

    def by_class(self, learned_labels=None):
        match = (self.predict == self.label).float()
        nlabels = int(max(torch.max(self.label).item(), torch.max(self.predict).item()))
        bc = {}
        ag = 0; ad = 0; am = 0
        for label in range(1, nlabels+1):
            lg = (self.label==label); ld = (self.predict==label)
            lr = torch.sum(match[lg]) / torch.sum(lg.float())
            lp = torch.sum(match[ld]) / torch.sum(ld.float())
            lf = 2 * lr * lp / (lr + lp)
            if torch.isnan(lf):
                bc[label] = (0, 0, 0)
            else:
                bc[label] = (lp.item(), lr.item(), lf.item())
            if learned_labels is not None and label in learned_labels:
                ag += lg.float().sum()
                ad += ld.float().sum()
                am += match[lg].sum()
        if learned_labels is None:
            ag = (self.label!=0); ad = (self.predict!=0)
            sum_ad = torch.sum(ag.float())
            if sum_ad == 0:
                ap = ar = 0
            else:
                ar = torch.sum(match[ag]) / torch.sum(ag.float())
                ap = torch.sum(match[ad]) / torch.sum(ad.float())
        else:
            if ad == 0:
                ap = ar = 0
            else:
                ar = am / ag; ap = am / ad
        if ap == 0:
            af = ap = ar = 0
        else:
            af = 2 * ar * ap / (ar + ap)
            af = af.item(); ar = ar.item(); ap = ap.item()
        return bc, (ap, ar, af)