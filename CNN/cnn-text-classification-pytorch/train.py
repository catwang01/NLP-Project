import torch

def train_one_epoch(model, dataiter, optimizer, criterion, print_interval=20):
    model.train()
    for step, batch in enumerate(dataiter, start=1):
        text = batch.text
        label = batch.label
        logits = model(text)
        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (torch.argmax(logits, dim=1) == label).float().mean()
        if step % print_interval == 0:
            print("Step: {}/{} train Loss: {} train Acc: {}".format(step,
                                                                    len(dataiter),
                                                                    loss.item(),
                                                                    acc.item()))

def eval_one_epoch(model, dataiter, criterion, best_loss, save_path="model.pkl"):
    model.eval()
    epoch_loss = 0.0
    epoch_correct = 0
    total = 0
    with torch.no_grad():
        for step, batch in enumerate(dataiter, start=1):
            text = batch.text
            label = batch.label
            logits = model(text)
            loss = criterion(logits, label)
            epoch_loss += loss.item()
            epoch_correct += (torch.argmax(logits, dim=1) == label).float().sum().item()
            total += label.size(0)
        print("Validation: Val Loss: {}, Val Acc: {}".format(epoch_loss / total,
                                                             epoch_correct / total))
        if best_loss > epoch_loss:
            torch.save(model, save_path)
            best_loss = epoch_loss
    return best_loss

def train_model(nepochs, model, trainiter, valiter, optimizer, criterion):
    best_loss = float("inf")
    for epoch in range(nepochs):
        print("Epoch: {}/{}".format(epoch, nepochs))
        train_one_epoch(model, trainiter, optimizer, criterion)
        best_loss = eval_one_epoch(model, valiter, criterion, best_loss)
