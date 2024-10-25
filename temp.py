import torch
logits = torch.tensor([[-1.3437e-01, -4.4162e-01, -1.6611e-02,  3.2343e-01,  1.1488e-01,
         -3.4555e-01, -2.0409e-02,  2.2216e-01, -2.7944e-01, -2.2451e-01,
          7.2233e-01,  2.8186e-01, -1.5752e-01, -4.5072e-02, -7.6948e-01,
         -3.9507e-01],
        [-3.2748e-01, -5.4203e-01, -3.8288e-01,  3.4406e-01,  1.9057e-01,
         -4.2266e-02, -1.3873e-01, -2.1568e-02, -2.8504e-01, -5.0806e-01,
          1.0452e-01,  5.3005e-01, -5.4518e-01,  5.7795e-02, -4.7876e-01,
         -5.8376e-01],])


topk=(1, 5)
labels = torch.tensor([0,1])
max_k = max(topk)
batch_size = labels.size(0)

_, pred = logits.topk(max_k, 1, True, True)
print(pred)
pred = pred.t()
print(pred)
print(labels.view(1, -1))
print(labels.view(1, -1).expand_as(pred))
print(pred.eq(labels.view(1, -1).expand_as(pred)))
matches = pred.eq(labels.view(1, -1).expand_as(pred))

res = []
for k in topk:
    matches_k = matches[:k].reshape(-1).float().sum(0, keepdim=True)
    wrong_k = batch_size - matches_k
    res.append(matches_k.mul_(100.0 / batch_size))

