import numpy as np

# System parameters
NUM_TX = 2              # Transmit antennas
NUM_RX = 2              # Receive antennas
NUM_SUBCARRIERS = 64    # OFDM subcarriers
NUM_OFDM_SYMBOLS = 14   # Number of symbols per frame
PILOT_SPACING = 4       # Every 4th subcarrier is a pilot
SNR_dB = 20             # Signal to Noise Ratio in dB

def qpsk_mod(bits):
    symbols = (2*bits[0::2]-1) + 1j*(2*bits[1::2]-1)
    return symbols / np.sqrt(2)

def generate_pilots(num_symbols, num_subcarriers, pilot_spacing):
    pilot_mask = np.zeros((num_symbols, num_subcarriers))
    pilot_mask[:, ::pilot_spacing] = 1
    return pilot_mask

def generate_mimo_channel(tx, rx):
    return (np.random.randn(rx, tx) + 1j * np.random.randn(rx, tx)) / np.sqrt(2)

def add_awgn(signal, snr_db):
    snr_linear = 10**(snr_db / 10)
    power = np.mean(np.abs(signal)**2)
    noise_power = power / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape))
    return signal + noise

# Simulate a batch
def simulate_batch(batch_size=1000):
    X_ls = []
    Y_true = []
    
    for _ in range(batch_size):
        tx_bits = np.random.randint(0, 2, (NUM_TX, NUM_OFDM_SYMBOLS * NUM_SUBCARRIERS * 2))
        tx_symbols = np.array([qpsk_mod(bits) for bits in tx_bits])
        tx_symbols = tx_symbols.reshape(NUM_TX, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS)

        channel = np.stack([generate_mimo_channel(NUM_TX, NUM_RX) for _ in range(NUM_OFDM_SYMBOLS * NUM_SUBCARRIERS)])
        channel = channel.reshape(NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS, NUM_RX, NUM_TX) 

        rx_symbols = np.zeros((NUM_RX, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS), dtype=complex)

        for sym_idx in range(NUM_OFDM_SYMBOLS):
            for sc_idx in range(NUM_SUBCARRIERS):
                h = channel[sym_idx, sc_idx]  # shape: RX x TX
                x = tx_symbols[:, sym_idx, sc_idx]
                rx_symbols[:, sym_idx, sc_idx] = h @ x

        rx_symbols_noisy = add_awgn(rx_symbols, SNR_dB)

        # LS Estimation at pilot positions only
        pilots = generate_pilots(NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS, PILOT_SPACING)
        ls_est = np.zeros((NUM_RX, NUM_TX, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS), dtype=complex)

        for sym_idx in range(NUM_OFDM_SYMBOLS):
            for sc_idx in range(0, NUM_SUBCARRIERS, PILOT_SPACING):
                if pilots[sym_idx, sc_idx] == 1:
                    y = rx_symbols_noisy[:, sym_idx, sc_idx]
                    x = tx_symbols[:, sym_idx, sc_idx]
                    H_ls = np.outer(y, np.conj(x)) / (np.abs(x) ** 2).sum()
                    ls_est[:, :, sym_idx, sc_idx] = H_ls

        X_ls.append(ls_est)
        Y_true.append(channel.transpose(2, 3, 0, 1))  # reshape to match output shape

    return np.array(X_ls), np.array(Y_true)

# Example usage
if __name__ == "__main__":
    X, Y = simulate_batch(100)
    print("LS Estimates shape:", X.shape)
    print("True Channel shape:", Y.shape)
