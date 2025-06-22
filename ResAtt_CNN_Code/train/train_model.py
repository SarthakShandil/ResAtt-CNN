from torch.utils.data import DataLoader, TensorDataset

def train_model(X_train, Y_train, num_epochs=10, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResAttCNN(in_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = mse_loss()
    loss_history = []

    def preprocess(X):
        real = X.real
        imag = X.imag
        return torch.tensor(np.stack([real, imag], axis=1), dtype=torch.float32)

    X_tensor = preprocess(X_train).reshape(-1, 2, 14, 64)
    Y_tensor = preprocess(Y_train).reshape(-1, 2, 14, 64)

    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f}")
        loss_history.append(avg_loss)

    return model, loss_history

