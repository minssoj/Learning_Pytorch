import torch

w = torch.tensor(2., requires_grad=True)
y = 2 * w
y.backward()

print('w로 미분한 값:', w.grad)

x = torch.tensor([2., 3.], requires_grad=True)
w = x ** 2
y = 2 * w + 4
target = torch.tensor([3., 4.])
loss = torch.sum(torch.abs(y - target))
loss.backward()

print('x로 미분한 값:', x.grad)
print(loss)
print(loss.item())

w = torch.tensor(2., requires_grad=True)
for epoch in range(20):
    y = 2 * w
    y.backward()
    print('w로 미분한 값:', w.grad)
    w.grad.zero_()
    #optimizer.zero_grad()
