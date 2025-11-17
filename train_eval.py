# train_eval.py
import torch
import torch.nn as nn
import torch.optim as optim
import copy

def train_and_eval(model, train_loader, test_loader, device, optimizer=None, num_epochs=1000):
    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_test_acc = 0.0
    best_model_path = 'subhajit_best_vit_classifier.pt'
    train_history = []
    test_history = []
    for epoch in range(1, num_epochs+1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        train_loss_avg = train_loss / train_total
        train_acc = 100. * train_correct / train_total
        train_history.append((train_loss_avg, train_acc))
        # Eval mode
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                test_loss += loss.item() * images.size(0)
                _, preds = torch.max(logits, 1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
        test_loss_avg = test_loss / test_total
        test_acc = 100. * test_correct / test_total
        test_history.append((test_loss_avg, test_acc))
        print(f"Epoch {epoch}: Train loss {train_loss_avg:.4f}, Train acc {train_acc:.2f}% | Test loss {test_loss_avg:.4f}, Test acc {test_acc:.2f}%")
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"[BEST SAVED] Epoch {epoch}, Test Acc: {test_acc:.2f}%")
    # Load and return best model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    print(f"Best model loaded with test accuracy: {best_test_acc:.2f}%")
    return model, train_history, test_history
