import numpy as np

NUM_TX = 2
NUM_RX = 2
NUM_SUBCARRIERS = 64
NUM_OFDM_SYMBOLS = 14
PILOT_SPACING = 4
SNR_dB = 20

def qpsk_mod(bits):
    symbols = (2*bits[0::2]-1) + 1j*(2*bits[1::2]-1)
    return symbols / np.sqrt(2)

def generate_pilots(num_symbols, num_subcarriers, spacing):
    mask = np.zeros((num_symbols, num_subcarriers))
    mask[:, ::spacing] = 1
    return mask

def generate_mimo_channel(tx, rx):
    return (np.random.randn(rx, tx) + 1j * np.random.randn(rx, tx)) / np.sqrt(2)

def add_awgn(signal, snr_db):
    snr_linear = 10**(snr_db / 10)
    power = np.mean(np.abs(signal)**2)
    noise_power = power / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape))
    return signal + noise

def simulate_batch(batch_size=100):
    X_ls = []
    Y_true = []

    for _ in range(batch_size):
        tx_bits = np.random.randint(0, 2, (NUM_TX, NUM_OFDM_SYMBOLS * NUM_SUBCARRIERS * 2))
        tx_symbols = np.array([qpsk_mod(bits) for bits in tx_bits]).reshape(NUM_TX, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS)

        channel = np.stack([generate_mimo_channel(NUM_TX, NUM_RX) for _ in range(NUM_OFDM_SYMBOLS * NUM_SUBCARRIERS)])
        channel = channel.reshape(NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS, NUM_RX, NUM_TX)

        rx_symbols = np.zeros((NUM_RX, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS), dtype=complex)

        for sym in range(NUM_OFDM_SYMBOLS):
            for sc in range(NUM_SUBCARRIERS):
                h = channel[sym, sc]
                x = tx_symbols[:, sym, sc]
                rx_symbols[:, sym, sc] = h @ x

        rx_noisy = add_awgn(rx_symbols, SNR_dB)

        pilots = generate_pilots(NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS, PILOT_SPACING)
        ls_est = np.zeros((NUM_RX, NUM_TX, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS), dtype=complex)

        for sym in range(NUM_OFDM_SYMBOLS):
            for sc in range(0, NUM_SUBCARRIERS, PILOT_SPACING):
                if pilots[sym, sc] == 1:
                    y = rx_noisy[:, sym, sc]
                    x = tx_symbols[:, sym, sc]
                    H_ls = np.outer(y, np.conj(x)) / (np.abs(x) ** 2).sum()
                    ls_est[:, :, sym, sc] = H_ls

        X_ls.append(ls_est)
        Y_true.append(channel.transpose(2, 3, 0, 1))

    return np.array(X_ls), np.array(Y_true)

