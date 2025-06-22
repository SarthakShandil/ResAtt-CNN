# Simulate data
X, Y = simulate_batch(batch_size=100)

# Train
model, loss_history = train_model(X, Y, num_epochs=10, batch_size=8)

# Evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def to_tensor(data):
    real = data.real
    imag = data.imag
    return torch.tensor(np.stack([real, imag], axis=1), dtype=torch.float32)

X_tensor = to_tensor(X).reshape(-1, 2, 14, 64).to(device)
Y_tensor = to_tensor(Y).reshape(-1, 2, 14, 64).to(device)


model.eval()
with torch.no_grad():
    pred = model(X_tensor).cpu()
    loss_val = nmse_loss(pred, Y_tensor.cpu())
    print(f"Final NMSE: {loss_val.item():.6f}")

# Plot results
plot_loss_curve(loss_history)

true_sample = Y[0, 0, 0]
pred_sample = pred[0, 0].numpy() + 1j * pred[0, 1].numpy()
plot_heatmap(true_sample, pred_sample)


