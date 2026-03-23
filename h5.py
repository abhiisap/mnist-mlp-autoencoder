import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ==============================================================
# Shared configuration
# ==============================================================

IMAGE_FILE = 'MNISTnumImages5000_balanced.txt'
LABEL_FILE = 'MNISTnumLabels5000_balanced.txt'

NUM_CLASSES = 10
SAMPLES_PER_CLASS = 500
TRAIN_SAMPLES_PER_CLASS = 400
TEST_SAMPLES_PER_CLASS = 100

# Hyperparameters (same as HW4)
HIDDEN_UNITS = 150
LEARNING_RATE = 0.1
MOMENTUM = 0.9
EPOCHS = 500
BATCH_SIZE = 64
L_THRESH = 0.75
H_THRESH = 0.25

# Output directory
HW5_REPORT_DIR = 'hw5_figs'
os.makedirs(HW5_REPORT_DIR, exist_ok=True)

# ==============================================================
# Basic utilities
# ==============================================================

def one_hot_encode(Y_int, num_classes=10):
    m = Y_int.shape[0]
    oh = np.zeros((m, num_classes))
    oh[np.arange(m), Y_int] = 1
    return oh


def load_and_split_data(image_file, label_file):
    """
    Same stratified split as HW4: 400 train + 100 test per class.
    """
    print(f"Loading data from '{image_file}' and '{label_file}'...")

    if not (os.path.exists(image_file) and os.path.exists(label_file)):
        raise FileNotFoundError("MNIST subset files not found in current directory.")

    X_all = np.loadtxt(image_file)
    Y_all_int = np.loadtxt(label_file, dtype=int)

    X_train_list, Y_train_list = [], []
    X_test_list, Y_test_list = [], []

    for digit in range(NUM_CLASSES):
        idx = np.where(Y_all_int == digit)[0]
        if len(idx) < SAMPLES_PER_CLASS:
            raise ValueError(f"Class {digit} has only {len(idx)} samples.")

        train_idx = idx[:TRAIN_SAMPLES_PER_CLASS]
        test_idx = idx[TRAIN_SAMPLES_PER_CLASS:TRAIN_SAMPLES_PER_CLASS + TEST_SAMPLES_PER_CLASS]

        X_train_list.append(X_all[train_idx])
        Y_train_list.append(Y_all_int[train_idx])
        X_test_list.append(X_all[test_idx])
        Y_test_list.append(Y_all_int[test_idx])

    X_train = np.vstack(X_train_list)
    Y_train_int = np.hstack(Y_train_list)
    X_test = np.vstack(X_test_list)
    Y_test_int = np.hstack(Y_test_list)

    print("Stratified train/test split complete.")
    print(f"Training set: X {X_train.shape}, Y {Y_train_int.shape}")
    print(f"Test set:     X {X_test.shape}, Y {Y_test_int.shape}")
    return X_train, Y_train_int, X_test, Y_test_int


def compute_confusion_matrix(Y_true_int, Y_pred_int, num_classes=10):
    mat = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(Y_true_int)):
        mat[Y_true_int[i], Y_pred_int[i]] += 1
    return mat


def per_digit_error_from_confusion(cm):
    """
    Given a confusion matrix (rows = true, cols = predicted),
    return a vector of per-digit error fractions.
    """
    counts = cm.sum(axis=1)
    errors = np.zeros(NUM_CLASSES)
    for k in range(NUM_CLASSES):
        if counts[k] > 0:
            acc_k = cm[k, k] / counts[k]
            errors[k] = 1.0 - acc_k
        else:
            errors[k] = np.nan
    return errors


def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, str(cm[i, j]),
                     ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black',
                     fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to '{save_path}'")
    plt.close()


def plot_error_history(epoch_log, train_errors, test_errors, title, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_log, train_errors, 'b-o', markersize=3, label='Train error')
    plt.plot(epoch_log, test_errors, 'r-o', markersize=3, label='Test error')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Error fraction')
    plt.grid(True)
    plt.legend()
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Error curve saved to '{save_path}'")
    plt.close()


def plot_overall_error_bars(train_error_final, test_error_final, title, save_path):
    """
    Bar plot of final mean training and test error (overall).
    """
    plt.figure(figsize=(5, 4))
    labels = ['Train', 'Test']
    values = [train_error_final, test_error_final]
    x = np.arange(len(labels))

    bars = plt.bar(x, values)
    plt.xticks(x, labels)
    plt.ylabel('Error fraction')
    plt.ylim(0.0, 1.0)
    plt.title(title)

    for bar, v in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0,
                 height + 0.01,
                 f'{v:.3f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Overall error bar plot saved to '{save_path}'")
    plt.close()


def plot_per_digit_error_bars(train_errors, test_errors, title, save_path):
    """
    Per-digit bar plot: for each digit 0–9, show train & test error bars.
    """
    digits = np.arange(NUM_CLASSES)
    width = 0.35

    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.xlabel('Digit')
    plt.ylabel('Error fraction')
    plt.ylim(0.0, 1.0)

    plt.bar(digits - width/2, train_errors, width, label='Train')
    plt.bar(digits + width/2, test_errors, width, label='Test')

    plt.xticks(digits, digits)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Per-digit error bar plot saved to '{save_path}'")
    plt.close()


def compute_threshold_accuracy(model, X, Y_int, L=L_THRESH, H=H_THRESH):
    """
    L/H threshold rule as in HW4.
    """
    A_L, _ = model._feed_forward(X.T)
    m = X.shape[0]
    correct = 0
    for i in range(m):
        true_label = Y_int[i]
        outputs = A_L[:, i]
        if outputs[true_label] >= L and np.all(np.delete(outputs, true_label) <= H):
            correct += 1
    return correct / m

# ==============================================================
# Generic MLP (same style as HW4, with optional frozen layers)
# ==============================================================

class MLPNetwork:
    def __init__(self, layer_dims, learning_rate=0.1, momentum=0.9):
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.weights = {}
        self.biases = {}
        self.velocities = {}

        for l in range(1, self.num_layers):
            n_out = self.layer_dims[l]
            n_in = self.layer_dims[l - 1]
            self.weights[l] = np.random.randn(n_out, n_in) * np.sqrt(1.0 / n_in)
            self.biases[l] = np.zeros((n_out, 1))
            self.velocities[f'W{l}'] = np.zeros((n_out, n_in))
            self.velocities[f'b{l}'] = np.zeros((n_out, 1))

    @staticmethod
    def _sigmoid(z):
        z_clip = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clip))

    @staticmethod
    def _sigmoid_derivative(a):
        return a * (1.0 - a)

    def _feed_forward(self, X_batch):
        cache = {}
        A = X_batch
        cache['A0'] = A
        for l in range(1, self.num_layers):
            W = self.weights[l]
            b = self.biases[l]
            Z = W @ A + b
            A = self._sigmoid(Z)
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
        return A, cache

    def _back_propagate(self, Y_batch_target, cache):
        grads = {}
        m = Y_batch_target.shape[1]
        L = self.num_layers - 1

        A_L = cache[f'A{L}']
        delta = (A_L - Y_batch_target) * self._sigmoid_derivative(A_L)

        A_prev = cache[f'A{L-1}']
        grads[f'dW{L}'] = (1.0 / m) * (delta @ A_prev.T)
        grads[f'db{L}'] = (1.0 / m) * np.sum(delta, axis=1, keepdims=True)

        for l in reversed(range(1, L)):
            W_next = self.weights[l + 1]
            delta_next = delta
            A_l = cache[f'A{l}']
            delta = (W_next.T @ delta_next) * self._sigmoid_derivative(A_l)

            A_prev = cache[f'A{l-1}']
            grads[f'dW{l}'] = (1.0 / m) * (delta @ A_prev.T)
            grads[f'db{l}'] = (1.0 / m) * np.sum(delta, axis=1, keepdims=True)

        return grads

    def _update_parameters(self, gradients):
        for l in range(1, self.num_layers):
            dW = gradients[f'dW{l}']
            db = gradients[f'db{l}']

            self.velocities[f'W{l}'] = self.momentum * self.velocities[f'W{l}'] + self.learning_rate * dW
            self.velocities[f'b{l}'] = self.momentum * self.velocities[f'b{l}'] + self.learning_rate * db

            self.weights[l] -= self.velocities[f'W{l}']
            self.biases[l] -= self.velocities[f'b{l}']

    # --- classifier helpers ---

    def predict(self, X):
        A_L, _ = self._feed_forward(X.T)
        return np.argmax(A_L, axis=0)

    def compute_error_fraction(self, X, Y_int):
        y_hat = self.predict(X)
        return np.mean(y_hat != Y_int)

    def train_classifier(self, X_train, Y_train_int,
                         X_test, Y_test_int,
                         epochs, batch_size=64,
                         freeze_layers=None,
                         label="Classifier"):
        """
        If freeze_layers is not None, it should be a set/list of layer indices
        (1-based) whose weights/biases will not be updated.
        """
        if freeze_layers is None:
            freeze_layers = set()
        else:
            freeze_layers = set(freeze_layers)

        m_train = X_train.shape[0]
        epoch_log = [0]
        train_err = self.compute_error_fraction(X_train, Y_train_int)
        test_err = self.compute_error_fraction(X_test, Y_test_int)
        train_error_log = [train_err]
        test_error_log = [test_err]

        print(f"[{label}] Epoch 0: Train Error = {train_err*100:.2f}%, Test Error = {test_err*100:.2f}%")

        num_classes = self.layer_dims[-1]
        start_time = time.time()

        for e in range(1, epochs + 1):
            idx = np.random.permutation(m_train)
            X_shuff = X_train[idx]
            Y_shuff = Y_train_int[idx]

            for i in range(0, m_train, batch_size):
                X_batch = X_shuff[i:i + batch_size]
                Y_batch_int = Y_shuff[i:i + batch_size]

                m_b = Y_batch_int.shape[0]
                Y_batch_oh = np.zeros((m_b, num_classes))
                Y_batch_oh[np.arange(m_b), Y_batch_int] = 1.0

                Xb = X_batch.T
                Yb = Y_batch_oh.T

                A_L, cache = self._feed_forward(Xb)
                grads = self._back_propagate(Yb, cache)

                # zero gradients for frozen layers
                for l in freeze_layers:
                    grads[f'dW{l}'][:] = 0.0
                    grads[f'db{l}'][:] = 0.0

                self._update_parameters(grads)

            if e % 10 == 0:
                train_err = self.compute_error_fraction(X_train, Y_train_int)
                test_err = self.compute_error_fraction(X_test, Y_test_int)
                epoch_log.append(e)
                train_error_log.append(train_err)
                test_error_log.append(test_err)
                print(f"[{label}] Epoch {e}/{epochs}: Train Error = {train_err*100:.2f}%, Test Error = {test_err*100:.2f}%")

        elapsed = time.time() - start_time
        print(f"[{label}] Training complete in {elapsed:.2f} seconds.")
        return epoch_log, train_error_log, test_error_log

    # --- autoencoder helpers (not used in HW5, but kept for completeness) ---

    def compute_mrl(self, X):
        X_T = X.T
        A_L, _ = self._feed_forward(X_T)
        per_sample_loss = 0.5 * np.sum((A_L - X_T) ** 2, axis=0)
        return np.mean(per_sample_loss), per_sample_loss

# ==============================================================
# Problem 1 – Transfer learning from HW4 autoencoder
# ==============================================================

def run_problem1_transfer(X_train, Y_train_int, X_test, Y_test_int):
    """
    Use the HW4 Problem-2 autoencoder weights for the input->hidden layer
    and repeat HW4 Problem-1 classifier with:
      - Case I: hidden layer frozen, train output only
      - Case II: full backprop on both layers
    """
    print("\n=== HW5 Problem 1: Transfer learning from HW4 autoencoder ===")

    # 1) Load AE weights from HW4 (Problem 2)
    ae_file = 'hw4_autoencoder_artifacts.npz'
    if not os.path.exists(ae_file):
        raise FileNotFoundError(
            f"Required file '{ae_file}' not found. "
            "Make sure you saved the autoencoder weights in HW4."
        )

    data = np.load(ae_file, allow_pickle=True)
    ae_weights = data['final_weights'].item()
    ae_biases = data['final_biases'].item()
    W_ae_in_hidden = ae_weights[1]  # shape (150, 784)
    b_ae_hidden = ae_biases[1]      # shape (150, 1)

    print(f"Loaded HW4 AE weights: input->hidden shape {W_ae_in_hidden.shape}")

    # 2) Case I – AE hidden fixed, train output layer only
    print("\n=== HW5 Problem 1: Case I (Frozen AE hidden, train output only) ===")
    case1_net = MLPNetwork([784, HIDDEN_UNITS, NUM_CLASSES],
                           learning_rate=LEARNING_RATE,
                           momentum=MOMENTUM)

    # Initialize hidden weights from AE
    case1_net.weights[1] = W_ae_in_hidden.copy()
    case1_net.biases[1] = b_ae_hidden.copy()

    # Train with layer 1 frozen
    c1_epochs, c1_train_err, c1_test_err = case1_net.train_classifier(
        X_train, Y_train_int,
        X_test, Y_test_int,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        freeze_layers={1},
        label="Case I"
    )

    # Error time series
    plot_error_history(
        c1_epochs, c1_train_err, c1_test_err,
        title="Case I Error vs Epoch (AE hidden frozen)",
        save_path=os.path.join(HW5_REPORT_DIR, 'case1_error_curve.png')
    )

    # Overall bar plot (final mean errors)
    plot_overall_error_bars(
        train_error_final=c1_train_err[-1],
        test_error_final=c1_test_err[-1],
        title="Case I Final Mean Classification Error",
        save_path=os.path.join(HW5_REPORT_DIR, 'case1_overall_error_bars.png')
    )

    # Confusion matrices
    Y_train_pred_c1 = case1_net.predict(X_train)
    Y_test_pred_c1 = case1_net.predict(X_test)
    cm_train_c1 = compute_confusion_matrix(Y_train_int, Y_train_pred_c1, NUM_CLASSES)
    cm_test_c1 = compute_confusion_matrix(Y_test_int, Y_test_pred_c1, NUM_CLASSES)

    plot_confusion_matrix(cm_train_c1,
                          "Case I Train Confusion Matrix",
                          os.path.join(HW5_REPORT_DIR, 'case1_confusion_train.png'))
    plot_confusion_matrix(cm_test_c1,
                          "Case I Test Confusion Matrix",
                          os.path.join(HW5_REPORT_DIR, 'case1_confusion_test.png'))

    # Per-digit error bars (train & test)
    c1_train_digit_err = per_digit_error_from_confusion(cm_train_c1)
    c1_test_digit_err = per_digit_error_from_confusion(cm_test_c1)
    plot_per_digit_error_bars(
        c1_train_digit_err,
        c1_test_digit_err,
        title="Case I Per-Digit Classification Error",
        save_path=os.path.join(HW5_REPORT_DIR, 'case1_per_digit_error_bars.png')
    )

    # Threshold-based accuracy (for report discussion)
    c1_train_th = compute_threshold_accuracy(case1_net, X_train, Y_train_int)
    c1_test_th = compute_threshold_accuracy(case1_net, X_test, Y_test_int)

    print(f"[Case I] Final train error (WTA): {c1_train_err[-1]*100:.2f}%")
    print(f"[Case I] Final test  error (WTA): {c1_test_err[-1]*100:.2f}%")
    print(f"[Case I] Threshold-based train acc: {c1_train_th*100:.2f}%")
    print(f"[Case I] Threshold-based test  acc: {c1_test_th*100:.2f}%")

    # 3) Case II – AE init, but train both layers
    print("\n=== HW5 Problem 1: Case II (AE hidden init, full backprop) ===")
    case2_net = MLPNetwork([784, HIDDEN_UNITS, NUM_CLASSES],
                           learning_rate=LEARNING_RATE,
                           momentum=MOMENTUM)

    case2_net.weights[1] = W_ae_in_hidden.copy()
    case2_net.biases[1] = b_ae_hidden.copy()

    c2_epochs, c2_train_err, c2_test_err = case2_net.train_classifier(
        X_train, Y_train_int,
        X_test, Y_test_int,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        freeze_layers=None,
        label="Case II"
    )

    # Error time series
    plot_error_history(
        c2_epochs, c2_train_err, c2_test_err,
        title="Case II Error vs Epoch (AE hidden fine-tuned)",
        save_path=os.path.join(HW5_REPORT_DIR, 'case2_error_curve.png')
    )

    # Overall bar plot (final mean errors)
    plot_overall_error_bars(
        train_error_final=c2_train_err[-1],
        test_error_final=c2_test_err[-1],
        title="Case II Final Mean Classification Error",
        save_path=os.path.join(HW5_REPORT_DIR, 'case2_overall_error_bars.png')
    )

    # Confusion matrices
    Y_train_pred_c2 = case2_net.predict(X_train)
    Y_test_pred_c2 = case2_net.predict(X_test)
    cm_train_c2 = compute_confusion_matrix(Y_train_int, Y_train_pred_c2, NUM_CLASSES)
    cm_test_c2 = compute_confusion_matrix(Y_test_int, Y_test_pred_c2, NUM_CLASSES)

    plot_confusion_matrix(cm_train_c2,
                          "Case II Train Confusion Matrix",
                          os.path.join(HW5_REPORT_DIR, 'case2_confusion_train.png'))
    plot_confusion_matrix(cm_test_c2,
                          "Case II Test Confusion Matrix",
                          os.path.join(HW5_REPORT_DIR, 'case2_confusion_test.png'))

    # Per-digit error bars (train & test)
    c2_train_digit_err = per_digit_error_from_confusion(cm_train_c2)
    c2_test_digit_err = per_digit_error_from_confusion(cm_test_c2)
    plot_per_digit_error_bars(
        c2_train_digit_err,
        c2_test_digit_err,
        title="Case II Per-Digit Classification Error",
        save_path=os.path.join(HW5_REPORT_DIR, 'case2_per_digit_error_bars.png')
    )

    # Threshold-based accuracy
    c2_train_th = compute_threshold_accuracy(case2_net, X_train, Y_train_int)
    c2_test_th = compute_threshold_accuracy(case2_net, X_test, Y_test_int)

    print(f"[Case II] Final train error (WTA): {c2_train_err[-1]*100:.2f}%")
    print(f"[Case II] Final test  error (WTA): {c2_test_err[-1]*100:.2f}%")
    print(f"[Case II] Threshold-based train acc: {c2_train_th*100:.2f}%")
    print(f"[Case II] Threshold-based test  acc: {c2_test_th*100:.2f}%")

    # Save logs for report
    np.savez(
        os.path.join(HW5_REPORT_DIR, 'hw5_problem1_logs.npz'),
        case1_epochs=np.array(c1_epochs),
        case1_train_err=np.array(c1_train_err),
        case1_test_err=np.array(c1_test_err),
        case1_cm_train=cm_train_c1,
        case1_cm_test=cm_test_c1,
        case1_train_digit_err=c1_train_digit_err,
        case1_test_digit_err=c1_test_digit_err,
        case1_train_th=c1_train_th,
        case1_test_th=c1_test_th,
        case2_epochs=np.array(c2_epochs),
        case2_train_err=np.array(c2_train_err),
        case2_test_err=np.array(c2_test_err),
        case2_cm_train=cm_train_c2,
        case2_cm_test=cm_test_c2,
        case2_train_digit_err=c2_train_digit_err,
        case2_test_digit_err=c2_test_digit_err,
        case2_train_th=c2_train_th,
        case2_test_th=c2_test_th
    )

    return case1_net, case2_net

# ==============================================================
# Problem 2 – 12x12 SOFM (SOM)
# ==============================================================

def train_sofm(X_train,
               grid_size=12,
               eta0=0.1,
               radius0=6.0,
               epochs=50,
               tau_eta=20.0,
               tau_radius=20.0):
    """
    Trains a 12x12 SOM with 784-dim inputs.
    Returns weight matrix of shape (144, 784).
    """
    num_neurons = grid_size * grid_size
    num_features = X_train.shape[1]

    W = 0.01 * np.random.randn(num_neurons, num_features)

    positions = np.array([(i, j) for i in range(grid_size) for j in range(grid_size)])

    def grid_dist2(idx1, idx2):
        p1 = positions[idx1]
        p2 = positions[idx2]
        d = p1 - p2
        return d[0]**2 + d[1]**2

    print("\n=== HW5 Problem 2: Training SOFM (12x12) ===")
    m = X_train.shape[0]

    for e in range(epochs):
        eta = eta0 * np.exp(-e / tau_eta)
        radius = radius0 * np.exp(-e / tau_radius)
        radius2 = radius * radius

        idx = np.random.permutation(m)
        X_shuff = X_train[idx]

        for x in X_shuff:
            diffs = W - x
            dists2 = np.sum(diffs**2, axis=1)
            bmu_idx = np.argmin(dists2)

            for k in range(num_neurons):
                gd2 = grid_dist2(bmu_idx, k)
                h = np.exp(-gd2 / (2.0 * radius2)) if radius2 > 0 else 0.0
                if h > 1e-4:
                    W[k] += eta * h * (x - W[k])

        print(f"[SOFM] Epoch {e+1}/{epochs} completed (eta={eta:.4f}, radius={radius:.3f})")

    return W


def sofm_bmu_indices(W_som, X):
    num_samples = X.shape[0]
    bmus = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        x = X[i]
        diffs = W_som - x
        dists2 = np.sum(diffs**2, axis=1)
        bmus[i] = np.argmin(dists2)
    return bmus


def compute_sofm_activity_maps(W_som, X_test, Y_test_int, grid_size=12):
    activity_maps = {}
    bmus_all = sofm_bmu_indices(W_som, X_test)

    for digit in range(NUM_CLASSES):
        idx = np.where(Y_test_int == digit)[0]
        counts = np.zeros((grid_size, grid_size), dtype=float)
        for i in idx:
            b = bmus_all[i]
            r = b // grid_size
            c = b % grid_size
            counts[r, c] += 1.0
        counts /= float(len(idx))  # 100
        activity_maps[digit] = counts

    return activity_maps


def plot_sofm_activity_maps(activity_maps, grid_size=12, save_path=None):
    fig, axes = plt.subplots(2, 5, figsize=(14, 5))
    fig.suptitle("SOFM Winning-Fraction Activity Maps (Test Set)", fontsize=14)

    for d in range(NUM_CLASSES):
        ax = axes[d // 5, d % 5]
        ax.imshow(activity_maps[d], cmap='hot', interpolation='nearest')
        ax.set_title(f"Digit {d}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path is not None:
        plt.savefig(save_path)
        print(f"SOFM activity maps saved to '{save_path}'")
        plt.close()
    else:
        plt.show()


def plot_sofm_weight_grid(W_som, grid_size=12, save_path=None):
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    fig.suptitle("SOFM Prototypes (12x12 grid)", fontsize=14)

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            img = W_som[idx].reshape(28, 28)
            ax = axes[i, j]
            ax.imshow(img, cmap='gray')
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path is not None:
        plt.savefig(save_path)
        print(f"SOFM weight grid saved to '{save_path}'")
        plt.close()
    else:
        plt.show()


def run_problem2_sofm(X_train, Y_train_int, X_test, Y_test_int):
    print("\n=== HW5 Problem 2: SOFM (12x12) ===")
    W_som = train_sofm(
        X_train,
        grid_size=12,
        eta0=0.1,
        radius0=6.0,
        epochs=50,
        tau_eta=20.0,
        tau_radius=20.0
    )

    activity_maps = compute_sofm_activity_maps(W_som, X_test, Y_test_int, grid_size=12)
    plot_sofm_activity_maps(
        activity_maps,
        grid_size=12,
        save_path=os.path.join(HW5_REPORT_DIR, 'sofm_activity_maps.png')
    )

    plot_sofm_weight_grid(
        W_som,
        grid_size=12,
        save_path=os.path.join(HW5_REPORT_DIR, 'sofm_weight_grid.png')
    )

    np.savez(
        os.path.join(HW5_REPORT_DIR, 'sofm_weights_hw5.npz'),
        W_som=W_som
    )

    return W_som, activity_maps

# ==============================================================
# Problem 3 – Classifier on top of SOFM
# ==============================================================

def compute_sofm_one_hot_features(W_som, X, grid_size=12):
    bmus = sofm_bmu_indices(W_som, X)
    num_samples = X.shape[0]
    num_neurons = grid_size * grid_size
    H = np.zeros((num_samples, num_neurons))
    H[np.arange(num_samples), bmus] = 1.0
    return H


def run_problem3_sofm_classifier(W_som, X_train, Y_train_int, X_test, Y_test_int, grid_size=12):
    print("\n=== HW5 Problem 3: SOFM-based classifier ===")

    H_train = compute_sofm_one_hot_features(W_som, X_train, grid_size=grid_size)
    H_test = compute_sofm_one_hot_features(W_som, X_test, grid_size=grid_size)

    clf = MLPNetwork([grid_size * grid_size, NUM_CLASSES],
                     learning_rate=LEARNING_RATE,
                     momentum=MOMENTUM)

    epochs, train_err, test_err = clf.train_classifier(
        H_train, Y_train_int,
        H_test, Y_test_int,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        freeze_layers=None,
        label="SOFM-Classifier"
    )

    # Error time series
    plot_error_history(
        epochs, train_err, test_err,
        title="SOFM-based Classifier Error vs Epoch",
        save_path=os.path.join(HW5_REPORT_DIR, 'sofm_classifier_error_curve.png')
    )

    # Test confusion matrix
    Y_test_pred = clf.predict(H_test)
    cm_test = compute_confusion_matrix(Y_test_int, Y_test_pred, NUM_CLASSES)
    plot_confusion_matrix(
        cm_test,
        "SOFM-based Classifier Test Confusion Matrix",
        os.path.join(HW5_REPORT_DIR, 'sofm_classifier_confusion_test.png')
    )

    # SOFM-to-output weight maps (10 heatmaps)
    W_out = clf.weights[1]  # (10, 144)
    fig, axes = plt.subplots(2, 5, figsize=(14, 5))
    fig.suptitle("SOFM-to-Output Weight Maps (12x12 per digit)", fontsize=14)

    for d in range(NUM_CLASSES):
        ax = axes[d // 5, d % 5]
        weights_grid = W_out[d].reshape(grid_size, grid_size)
        ax.imshow(weights_grid, cmap='bwr', interpolation='nearest')
        ax.set_title(f"Digit {d}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(HW5_REPORT_DIR, 'sofm_output_weight_maps.png')
    plt.savefig(save_path)
    print(f"SOFM output weight maps saved to '{save_path}'")
    plt.close()

    print(f"[SOFM-Classifier] Final train error (WTA): {train_err[-1]*100:.2f}%")
    print(f"[SOFM-Classifier] Final test  error (WTA): {test_err[-1]*100:.2f}%")

    np.savez(
        os.path.join(HW5_REPORT_DIR, 'hw5_problem3_logs.npz'),
        epochs=np.array(epochs),
        train_err=np.array(train_err),
        test_err=np.array(test_err),
        cm_test=cm_test,
        W_out=W_out
    )

    return clf

# ==============================================================
# Main driver
# ==============================================================

if __name__ == "__main__":
    X_train, Y_train_int, X_test, Y_test_int = load_and_split_data(IMAGE_FILE, LABEL_FILE)

    # Problem 1: transfer learning from HW4 AE
    case1_net, case2_net = run_problem1_transfer(X_train, Y_train_int, X_test, Y_test_int)

    # Problem 2: SOFM
    W_som, activity_maps = run_problem2_sofm(X_train, Y_train_int, X_test, Y_test_int)

    # Problem 3: SOFM-based classifier
    som_clf = run_problem3_sofm_classifier(W_som, X_train, Y_train_int, X_test, Y_test_int, grid_size=12)

    print("\nAll HW5 experiments completed. Figures and logs are in the 'hw5_figs' directory.")


