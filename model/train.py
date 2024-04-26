# train.py

def train_model(model, train_loader, val_loader, optimizer, criterion, device, scheduler=None, use_scheduler=False, epochs=10, patience=3, checkpoint_path='best_model_checkpoint.pth', use_early_stopping=True, for_val_loss_only=False):
    model.to(device)

    # Early stopping and best model parameters
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # Initialize lists to store per-epoch loss and accuracy for both training and validation
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Training Loss: {train_loss:.6f} Training Accuracy: {train_acc:.6f}')

        # Validation Phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Validation Loss: {val_loss:.6f} Validation Accuracy: {val_acc:.6f}')

        # Checkpoint and Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Update best model state
            patience_counter = 0
        else:
            if use_early_stopping:
                patience_counter += 1

                if patience_counter >= patience:
                    print(f'Early stopping triggered at epoch {epoch + 1}')
                    break

        # Update scheduler if it's being used
        if use_scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

    # save the best model here
    if best_model_state is not None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
        }, checkpoint_path)
        print(f'Best model saved: {best_val_loss}')

    if for_val_loss_only:
        return best_val_loss

    return train_losses, train_accs, val_losses, val_accs
