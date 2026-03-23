import numpy as np
import matplotlib.pyplot as plt
import time
import os
import seaborn as sns          # used only for the similarity heatmap
from numpy.linalg import norm  # used for cosine similarity

# ------------------------------------------------------------------
# Shared configuration
# ------------------------------------------------------------------
IMAGE_FILE = 'MNISTnumImages5000_balanced.txt'
LABEL_FILE = 'MNISTnumLabels5000_balanced.txt'

NUM_CLASSES = 10
SAMPLES_PER_CLASS = 500
TRAIN_SAMPLES_PER_CLASS = 400
TEST_SAMPLES_PER_CLASS = 100

# Problem 1 outputs
ARTIFACT_FILE = 'homework_4_artifacts.npz'        # keep for HW5 compatibility
PLOT_FILE = 'clf_error_curve.png'                 # classifier error plot
CLF_CONF_TRAIN_FILE = 'clf_confusion_train.png'   # train confusion matrix image
CLF_CONF_TEST_FILE = 'clf_confusion_test.png'     # test confusion matrix image

# Problem 2 outputs
P1_ARTIFACT_FILE = 'homework_4_artifacts.npz'     # weights from Problem 1
REPORT_DIR = 'hw4_autoencoder_figs'
F_MRL_HISTORY_PLOT = os.path.join(REPORT_DIR, 'ae_mrl_history.png')
F_OVERALL_MRL_PLOT = os.path.join(REPORT_DIR, 'ae_overall_mrl.png')
F_AE_FEATURES_PLOT = os.path.join(REPORT_DIR, 'ae_hidden_features.png')
F_CLF_FEATURES_PLOT = os.path.join(REPORT_DIR, 'clf_hidden_features_match.png')
F_SIM_HEATMAP_PLOT = os.path.join(REPORT_DIR, 'ae_feature_sim_matrix.png')
F_SIM_BAR_PLOT = os.path.join(REPORT_DIR, 'ae_feature_maxsim_bar.png')
F_RECONSTRUCTION_PLOT = os.path.join(REPORT_DIR, 'ae_sample_recons.png')

# global hyperparams (set inside run_problem1 / run_problem2)
LAYER_DIMENSIONS = None
LEARNING_RATE = None
MOMENTUM = None
EPOCHS = None
BATCH_SIZE = None

# remember last hidden-layer configuration so Problem 2 can re-use it
LAST_HIDDEN_DIMS = None


# ------------------------------------------------------------------
# Basic nonlinearities
# ------------------------------------------------------------------

def _sigmoid(z):
    """
    Logistic activation.

    Input is clipped to avoid numerical issues in exp().
    """
    z_clip = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clip))


def _sigmoid_derivative(a):
    """
    Derivative of sigmoid, expressed in terms of the activation a.
    (a = sigmoid(z))
    """
    return a * (1.0 - a)


# ------------------------------------------------------------------
# Generic MLP class (used for both classifier and autoencoder)
# ------------------------------------------------------------------

class MLPNetwork:
    """
    Plain NumPy implementation of a fully-connected feedforward network.

    - Arbitrary layer sizes (layer_dims)
    - Sigmoid units everywhere
    - MSE loss with backprop
    - Momentum-based parameter updates
    """

    def __init__(self, layer_dims, learning_rate=0.1, momentum=0.9):
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.weights = {}
        self.biases = {}
        self.velocities = {}

        # parameter initialization
        for l in range(1, self.num_layers):
            n_out = self.layer_dims[l]
            n_in = self.layer_dims[l - 1]

            # Xavier/Glorot normal init – works reasonably with sigmoid
            self.weights[l] = np.random.randn(n_out, n_in) * np.sqrt(1.0 / n_in)
            self.biases[l] = np.zeros((n_out, 1))

            # momentum buffers
            self.velocities[f'W{l}'] = np.zeros((n_out, n_in))
            self.velocities[f'b{l}'] = np.zeros((n_out, 1))

    # ---------- forward / backward / update ----------

    def _feed_forward(self, X_batch):
        """
        Forward pass.

        X_batch: (n_features, batch_size)
        Returns:
          A_L  : output activations
          cache: intermediate A / Z values for backprop
        """
        cache = {}
        A = X_batch
        cache['A0'] = A

        for l in range(1, self.num_layers):
            W = self.weights[l]
            b = self.biases[l]
            Z = W @ A + b
            A = _sigmoid(Z)

            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A

        return A, cache

    def _back_propagate(self, Y_batch_target, cache):
        """
        Backprop for MSE loss.

        Y_batch_target:
          - classifier: one-hot labels
          - autoencoder: original input
        Both are shaped (n_out, batch_size).
        """
        grads = {}
        m = Y_batch_target.shape[1]
        L = self.num_layers - 1

        # output layer delta
        A_L = cache[f'A{L}']
        delta = (A_L - Y_batch_target) * _sigmoid_derivative(A_L)

        # gradients at output layer
        A_prev = cache[f'A{L-1}']
        grads[f'dW{L}'] = (1.0 / m) * (delta @ A_prev.T)
        grads[f'db{L}'] = (1.0 / m) * np.sum(delta, axis=1, keepdims=True)

        # propagate backwards through hidden layers
        for l in reversed(range(1, L)):
            W_next = self.weights[l + 1]
            delta_next = delta
            A_l = cache[f'A{l}']

            delta = (W_next.T @ delta_next) * _sigmoid_derivative(A_l)

            A_prev = cache[f'A{l-1}']
            grads[f'dW{l}'] = (1.0 / m) * (delta @ A_prev.T)
            grads[f'db{l}'] = (1.0 / m) * np.sum(delta, axis=1, keepdims=True)

        return grads

    def _update_parameters(self, gradients):
        """
        Apply momentum update to all weights and biases.
        """
        for l in range(1, self.num_layers):
            dW = gradients[f'dW{l}']
            db = gradients[f'db{l}']

            self.velocities[f'W{l}'] = self.momentum * self.velocities[f'W{l}'] + self.learning_rate * dW
            self.velocities[f'b{l}'] = self.momentum * self.velocities[f'b{l}'] + self.learning_rate * db

            self.weights[l] -= self.velocities[f'W{l}']
            self.biases[l] -= self.velocities[f'b{l}']

    # ---------- classifier helpers (Problem 1) ----------

    def predict(self, X):
        """
        Winner-take-all prediction.

        X: (num_samples, num_features)
        Returns an array of integer class labels.
        """
        A_L, _ = self._feed_forward(X.T)
        return np.argmax(A_L, axis=0)

    def compute_error_fraction(self, X, Y_int):
        """
        Misclassification rate = 1 - accuracy, using winner-take-all.
        """
        y_hat = self.predict(X)
        num_wrong = np.sum(y_hat != Y_int)
        return num_wrong / X.shape[0]

    def train_classifier(self, X_train, Y_train_int, X_test, Y_test_int, epochs, batch_size=64):
        """
        Training loop for the digit classifier.
        """
        m_train = X_train.shape[0]
        epoch_log = []

        # error before any learning
        train_err = self.compute_error_fraction(X_train, Y_train_int)
        test_err = self.compute_error_fraction(X_test, Y_test_int)
        epoch_log.append(0)
        train_error_log = [train_err]
        test_error_log = [test_err]

        print(f"[Classifier] Epoch 0: Train Error = {train_err*100:.2f}%, Test Error = {test_err*100:.2f}%")

        start = time.time()

        for e in range(1, epochs + 1):
            # shuffle indices each epoch
            idx = np.random.permutation(m_train)
            X_train_shuff = X_train[idx]
            Y_train_shuff = Y_train_int[idx]

            # mini-batches
            for i in range(0, m_train, batch_size):
                X_batch = X_train_shuff[i:i + batch_size]
                Y_batch_int = Y_train_shuff[i:i + batch_size]

                Y_batch_one_hot = one_hot_encode(Y_batch_int, NUM_CLASSES)

                X_batch_T = X_batch.T
                Y_batch_one_hot_T = Y_batch_one_hot.T

                A_L, cache = self._feed_forward(X_batch_T)
                grads = self._back_propagate(Y_batch_one_hot_T, cache)
                self._update_parameters(grads)

            if e % 10 == 0:
                train_err = self.compute_error_fraction(X_train, Y_train_int)
                test_err = self.compute_error_fraction(X_test, Y_test_int)

                epoch_log.append(e)
                train_error_log.append(train_err)
                test_error_log.append(test_err)

                print(f"[Classifier] Epoch {e}/{epochs}: "
                      f"Train Error = {train_err*100:.2f}%, Test Error = {test_err*100:.2f}%")

        elapsed = time.time() - start
        print(f"\n[Classifier] Training complete in {elapsed:.2f} seconds.")

        return epoch_log, train_error_log, test_error_log

    # ---------- autoencoder helpers (Problem 2) ----------

    def compute_mrl(self, X):
        """
        Mean reconstruction loss (MSE with 1/2 factor) over all samples.
        """
        X_T = X.T
        A_L, _ = self._feed_forward(X_T)
        per_sample_loss = 0.5 * np.sum((A_L - X_T) ** 2, axis=0)
        return np.mean(per_sample_loss), per_sample_loss

    def reconstruct(self, X):
        """
        Forward pass returning reconstruction with same shape as X.
        """
        A_L, _ = self._feed_forward(X.T)
        return A_L.T

    def autoencoder_train(self, X_train, X_test, epochs, batch_size=64):
        """
        Training loop for the autoencoder.
        """
        m_train = X_train.shape[0]

        epoch_log = []
        train_mrl_log = []
        test_mrl_log = []

        train_mrl, _ = self.compute_mrl(X_train)
        test_mrl, _ = self.compute_mrl(X_test)

        epoch_log.append(0)
        train_mrl_log.append(train_mrl)
        test_mrl_log.append(test_mrl)

        print(f"[Autoencoder] Epoch 0: Train MRL = {train_mrl:.6f}, Test MRL = {test_mrl:.6f}")

        start = time.time()

        for e in range(1, epochs + 1):
            idx = np.random.permutation(m_train)
            X_train_shuff = X_train[idx]

            for i in range(0, m_train, batch_size):
                X_batch = X_train_shuff[i:i + batch_size]
                X_batch_T = X_batch.T

                A_L, cache = self._feed_forward(X_batch_T)
                grads = self._back_propagate(X_batch_T, cache)
                self._update_parameters(grads)

            if e % 10 == 0:
                train_mrl, _ = self.compute_mrl(X_train)
                test_mrl, _ = self.compute_mrl(X_test)

                epoch_log.append(e)
                train_mrl_log.append(train_mrl)
                test_mrl_log.append(test_mrl)

                print(f"[Autoencoder] Epoch {e}/{epochs}: "
                      f"Train MRL = {train_mrl:.6f}, Test MRL = {test_mrl:.6f}")

        elapsed = time.time() - start
        print(f"\n[Autoencoder] Training complete in {elapsed:.2f} seconds.")

        return epoch_log, train_mrl_log, test_mrl_log


# ------------------------------------------------------------------
# Utility functions shared by both problems
# ------------------------------------------------------------------

def one_hot_encode(Y_int, num_classes=10):
    """
    Integer labels -> one-hot matrix.
    """
    m = Y_int.shape[0]
    oh = np.zeros((m, num_classes))
    oh[np.arange(m), Y_int] = 1
    return oh


def load_and_split_data(image_file, label_file):
    """
    Load the 5000 MNIST subset and create 400/100 per-digit split.
    """
    print(f"Loading data from '{image_file}' and '{label_file}'...")

    if not (os.path.exists(image_file) and os.path.exists(label_file)):
        print("!! Data files not found in current directory.")
        return None, None, None, None

    try:
        X_all = np.loadtxt(image_file)
        Y_all_int = np.loadtxt(label_file, dtype=int)
    except IOError as e:
        print(f"Error loading data files: {e}")
        return None, None, None, None

    if X_all.shape[0] != SAMPLES_PER_CLASS * NUM_CLASSES:
        print(f"Unexpected number of samples (got {X_all.shape[0]}).")
        return None, None, None, None

    X_train_list, Y_train_list = [], []
    X_test_list, Y_test_list = [], []

    for digit in range(NUM_CLASSES):
        idx = np.where(Y_all_int == digit)[0]

        if len(idx) < SAMPLES_PER_CLASS:
            print(f"Warning: class {digit} has only {len(idx)} samples.")
            continue

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
    """
    Simple confusion matrix: rows = true class, cols = predicted class.
    """
    mat = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(Y_true_int)):
        mat[Y_true_int[i], Y_pred_int[i]] += 1
    return mat


def plot_confusion_matrix(cm, title, save_path):
    """
    Visualize a confusion matrix as an image and save it.
    """
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
    print(f"[Classifier] Confusion matrix saved to '{save_path}'")
    plt.close()


def plot_error_history(epoch_log, train_errors, test_errors, save_path=PLOT_FILE):
    """
    Plot classifier error fraction vs epoch.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_log, train_errors, 'b-o', label='Training Error', markersize=4)
    plt.plot(epoch_log, test_errors, 'r-o', label='Test Error', markersize=4)
    plt.title(f'Classifier Error vs Epoch (LR={LEARNING_RATE}, Momentum={MOMENTUM})')
    plt.xlabel('Epoch')
    plt.ylabel('Error Fraction (1.0 - Accuracy)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.0, 1.0)
    plt.savefig(save_path)
    print(f"\n[Classifier] Error curve saved to '{save_path}'")
    plt.close()


def compute_threshold_accuracy(model, X, Y_int, L=0.75, H=0.25):
    """
    Compute accuracy using L/H thresholds as discussed in class.

    For each sample:
      - Let t be the true class.
      - Prediction is counted correct if:
          output[t] >= L  and  all other outputs <= H.
      Otherwise counted incorrect.
    """
    A_L, _ = model._feed_forward(X.T)   # shape (NUM_CLASSES, num_samples)
    m = X.shape[0]
    correct = 0

    for i in range(m):
        true_label = Y_int[i]
        outputs = A_L[:, i]
        if outputs[true_label] >= L and np.all(np.delete(outputs, true_label) <= H):
            correct += 1

    return correct / m


def parse_hidden_layer_dims():
    """
    Ask the user for hidden layer sizes at run-time.

    Examples:
      '150'      -> [150]
      '150,100'  -> [150, 100]

    If input is invalid or empty, defaults to [150].
    """
    s = input("Enter hidden layer sizes (e.g., '150' or '150,100'; blank for 150): ").strip()
    if not s:
        print("No hidden size entered; using default [150].")
        return [150]

    try:
        dims = [int(x) for x in s.replace(' ', '').split(',') if x]
        if not dims or any(d <= 0 for d in dims):
            raise ValueError
        return dims
    except ValueError:
        print("Invalid format for hidden sizes; falling back to [150].")
        return [150]


# ------------------------------------------------------------------
# Problem 2 helper plots
# ------------------------------------------------------------------

def plot_mrl_history(epoch_log, train_mrls, test_mrls, save_path):
    """
    Plot training/test MRL over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_log, train_mrls, 'b-o', label='Training MRL', markersize=4)
    plt.plot(epoch_log, test_mrls, 'r-o', label='Test MRL', markersize=4)
    plt.title(f'Autoencoder MRL vs Epoch (LR={LEARNING_RATE}, Momentum={MOMENTUM})')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Reconstruction Loss (J2)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"\n[Autoencoder] MRL history saved to '{save_path}'")
    plt.close()


def plot_overall_performance(train_mrl, test_mrl, save_path):
    """
    Bar chart of final train vs test MRL.
    """
    plt.figure(figsize=(8, 6))
    labels = ['Training', 'Test']
    vals = [train_mrl, test_mrl]
    bars = plt.bar(labels, vals, color=['blue', 'red'])
    plt.ylabel('Mean Reconstruction Loss (J2)')
    plt.title('Final Autoencoder Reconstruction Error')
    plt.bar_label(bars, fmt='%.6f')
    plt.savefig(save_path)
    print(f"[Autoencoder] Overall MRL bar chart saved to '{save_path}'")
    plt.close()


def calculate_class_wise_mrl(per_sample_loss_test, Y_test_int):
    """
    Per-digit mean and std of reconstruction loss.
    """
    means = []
    stds = []

    for digit in range(NUM_CLASSES):
        idx = np.where(Y_test_int == digit)[0]
        losses = per_sample_loss_test[idx]

        means.append(np.mean(losses))
        stds.append(np.std(losses, ddof=0))

    return means, stds


def plot_neuron_features(weights, title, neuron_indices, save_path):
    """
    Show 20 weight vectors as 28x28 images in a 4x5 grid.
    """
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    fig.suptitle(title, fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i >= len(neuron_indices):
            ax.axis('off')
            continue

        idx = neuron_indices[i]
        img = weights[idx].reshape(28, 28)

        ax.imshow(img, cmap='gray')
        ax.set_title(f'Neuron {idx}')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"[Autoencoder] Feature grid saved to '{save_path}'")
    plt.close()


def plot_feature_similarities(weights, save_path_heatmap, save_path_bar):
    """
    Cosine-similarity heatmap and max-similarity bar plot.
    """
    print("Computing cosine similarities between hidden units...")

    norms = norm(weights, axis=1, keepdims=True)
    W_norm = weights / (norms + 1e-10)
    sim_matrix = W_norm @ W_norm.T

    # heatmap of cosine similarities
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, vmin=-1, vmax=1, cmap='vlag', square=True)
    plt.title('Autoencoder Hidden-Unit Cosine Similarities')
    plt.xlabel('Neuron Index')
    plt.ylabel('Neuron Index')
    plt.savefig(save_path_heatmap)
    print(f"[Autoencoder] Similarity matrix saved to '{save_path_heatmap}'")
    plt.close()

    # bar plot of highest non-diagonal similarity
    np.fill_diagonal(sim_matrix, -np.inf)
    max_sim = np.max(sim_matrix, axis=1)

    plt.figure(figsize=(12, 6))
    plt.bar(range(weights.shape[0]), max_sim)
    plt.title('Max Cosine Similarity per Hidden Neuron (Excluding Self)')
    plt.xlabel('Neuron Index')
    plt.ylabel('Max Cosine Similarity')
    plt.ylim(0.0, 1.0)
    plt.savefig(save_path_bar)
    print(f"[Autoencoder] Max-similarity bar plot saved to '{save_path_bar}'")
    plt.close()


def plot_sample_outputs(X_test, reconstructed_images, save_path):
    """
    Show 8 random original vs reconstructed images (2x8 grid).
    """
    num_images = 8
    idx = np.random.choice(X_test.shape[0], num_images, replace=False)

    fig, axes = plt.subplots(2, num_images, figsize=(16, 4))
    fig.suptitle('Autoencoder Sample Reconstructions', fontsize=16)

    for i, j in enumerate(idx):
        orig = X_test[j].reshape(28, 28)
        axes[0, i].imshow(orig, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Original")

        recon = reconstructed_images[j].reshape(28, 28)
        axes[1, i].imshow(recon, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("Reconstructed")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    print(f"[Autoencoder] Example reconstructions saved to '{save_path}'")
    plt.close()


# ------------------------------------------------------------------
# Problem-specific wrappers
# ------------------------------------------------------------------

def run_problem1():
    """
    Problem 1: classifier training and artifact dump.
    """
    global LAYER_DIMENSIONS, LEARNING_RATE, MOMENTUM, EPOCHS, BATCH_SIZE, LAST_HIDDEN_DIMS

    # Ask user for hidden-layer structure
    print("\n--- Problem 1: Hidden-layer configuration ---")
    hidden_dims = parse_hidden_layer_dims()
    LAST_HIDDEN_DIMS = hidden_dims[:]  # store for Problem 2

    LAYER_DIMENSIONS = [784] + hidden_dims + [10]
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    EPOCHS = 500
    BATCH_SIZE = 64

    data = load_and_split_data(IMAGE_FILE, LABEL_FILE)
    if data is None:
        return

    X_train, Y_train_int, X_test, Y_test_int = data

    print("\n=== DIGIT CLASSIFIER TRAINING (Problem 1) ===")
    print(f"Network architecture: {LAYER_DIMENSIONS}")
    print(f"Learning rate: {LEARNING_RATE}, Momentum: {MOMENTUM}, Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")

    nn = MLPNetwork(
        layer_dims=LAYER_DIMENSIONS,
        learning_rate=LEARNING_RATE,
        momentum=MOMENTUM
    )

    print(f"Running classifier training...")
    epoch_log, train_error_log, test_error_log = nn.train_classifier(
        X_train, Y_train_int,
        X_test, Y_test_int,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    plot_error_history(epoch_log, train_error_log, test_error_log)

    print("\n--- FINAL CLASSIFIER METRICS (winner-take-all) ---")
    Y_pred_train_int = nn.predict(X_train)
    Y_pred_test_int = nn.predict(X_test)

    train_err = train_error_log[-1]
    test_err = test_error_log[-1]
    train_acc = 1.0 - train_err
    test_acc = 1.0 - test_err
    print(f"Training accuracy (WTA): {train_acc * 100:.2f}% (error {train_err * 100:.2f}%)")
    print(f"Test accuracy     (WTA): {test_acc * 100:.2f}% (error {test_err * 100:.2f}%)")

    # Threshold-based metric using L/H
    print("\n--- L/H THRESHOLD-BASED ACCURACY ---")
    th_train_acc = compute_threshold_accuracy(nn, X_train, Y_train_int, L=0.75, H=0.25)
    th_test_acc = compute_threshold_accuracy(nn, X_test, Y_test_int, L=0.75, H=0.25)
    print(f"Training accuracy (L=0.75, H=0.25): {th_train_acc * 100:.2f}%")
    print(f"Test accuracy     (L=0.75, H=0.25): {th_test_acc * 100:.2f}%")

    # Confusion matrices
    cm_train = compute_confusion_matrix(Y_train_int, Y_pred_train_int, NUM_CLASSES)
    cm_test = compute_confusion_matrix(Y_test_int, Y_pred_test_int, NUM_CLASSES)

    print("\nTraining-set confusion matrix (rows=actual, cols=predicted):")
    print("Pred: 0   1   2   3   4   5   6   7   8   9")
    print("Act:")
    for i in range(NUM_CLASSES):
        print(f"{i}   {cm_train[i]}")

    print("\nTest-set confusion matrix (rows=actual, cols=predicted):")
    print("Pred: 0   1   2   3   4   5   6   7   8   9")
    print("Act:")
    for i in range(NUM_CLASSES):
        print(f"{i}   {cm_test[i]}")

    # Save confusion matrices as figures
    plot_confusion_matrix(cm_train, "Training Confusion Matrix", CLF_CONF_TRAIN_FILE)
    plot_confusion_matrix(cm_test, "Test Confusion Matrix", CLF_CONF_TEST_FILE)

    print("\n--- SAVING CLASSIFIER ARTIFACTS FOR HW5 ---")
    Y_test_one_hot = one_hot_encode(Y_test_int, NUM_CLASSES)
    A_L_test, _ = nn._feed_forward(X_test.T)
    Y_test_one_hot_T = Y_test_one_hot.T
    test_errors_per_sample = 0.5 * np.sum((A_L_test - Y_test_one_hot_T) ** 2, axis=0)

    hyperparameters = {
        'layer_dims': LAYER_DIMENSIONS,
        'learning_rate': LEARNING_RATE,
        'momentum': MOMENTUM,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE
    }

    try:
        np.savez(
            ARTIFACT_FILE,
            final_weights=nn.weights,
            final_biases=nn.biases,
            hyperparameters=hyperparameters,
            test_errors_per_sample=test_errors_per_sample
        )
        print(f"\nClassifier artifacts written to '{ARTIFACT_FILE}'")
        data = np.load(ARTIFACT_FILE, allow_pickle=True)
        print(f"  * Hidden layer weight shape: {data['final_weights'].item().get(1).shape}")
        print(f"  * Per-sample error vector:   {data['test_errors_per_sample'].shape}")
        print(f"  * Stored LR:                 {data['hyperparameters'].item().get('learning_rate')}")
    except Exception as e:
        print(f"Error while saving artifacts: {e}")

    print("\n>>> Problem 1 run finished.\n")


def run_problem2():
    """
    Problem 2: autoencoder training and analysis.
    """
    global LAYER_DIMENSIONS, LEARNING_RATE, MOMENTUM, EPOCHS, BATCH_SIZE, LAST_HIDDEN_DIMS

    os.makedirs(REPORT_DIR, exist_ok=True)

    # Use the same hidden-layer configuration as in Problem 1 if available,
    # otherwise ask the user again.
    if LAST_HIDDEN_DIMS is not None:
        hidden_dims = LAST_HIDDEN_DIMS[:]
        print("\n--- Problem 2: Re-using hidden-layer configuration from Problem 1 ---")
        print(f"Hidden layer sizes: {hidden_dims}")
    else:
        print("\n--- Problem 2: Hidden-layer configuration (same style as Problem 1) ---")
        hidden_dims = parse_hidden_layer_dims()
        LAST_HIDDEN_DIMS = hidden_dims[:]

    LAYER_DIMENSIONS = [784] + hidden_dims + [784]
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    EPOCHS = 500
    BATCH_SIZE = 64

    data = load_and_split_data(IMAGE_FILE, LABEL_FILE)
    if data is None:
        return

    X_train, Y_train_int, X_test, Y_test_int = data

    print("\n=== AUTOENCODER TRAINING (Problem 2) ===")
    print(f"Network architecture: {LAYER_DIMENSIONS}")
    print(f"Learning rate: {LEARNING_RATE}, Momentum: {MOMENTUM}, Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")

    ae = MLPNetwork(
        layer_dims=LAYER_DIMENSIONS,
        learning_rate=LEARNING_RATE,
        momentum=MOMENTUM
    )

    print(f"Running autoencoder training...")
    epoch_log, train_mrl_log, test_mrl_log = ae.autoencoder_train(
        X_train, X_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    print("\n--- AUTOENCODER SUMMARY STATISTICS ---")
    final_train_mrl = train_mrl_log[-1]
    final_test_mrl, per_sample_loss_test = ae.compute_mrl(X_test)
    print(f"Final training MRL: {final_train_mrl:.6f}")
    print(f"Final test MRL:     {final_test_mrl:.6f}")
    plot_overall_performance(final_train_mrl, final_test_mrl, F_OVERALL_MRL_PLOT)

    mrls, stds = calculate_class_wise_mrl(per_sample_loss_test, Y_test_int)
    print("\nPer-digit reconstruction statistics (test set):")
    print("| Digit | Mean MRL | Std Dev |")
    print("|-------|----------|---------|")
    for i in range(NUM_CLASSES):
        print(f"| {i:<5} | {mrls[i]:<8.6f} | {stds[i]:<7.6f} |")

    plot_mrl_history(epoch_log, train_mrl_log, test_mrl_log, F_MRL_HISTORY_PLOT)

    # feature comparison with classifier
    try:
        p1_data = np.load(P1_ARTIFACT_FILE, allow_pickle=True)
        p1_weights = p1_data['final_weights'].item().get(1)

        neuron_indices = np.random.choice(LAYER_DIMENSIONS[1], 20, replace=False)

        ae_weights = ae.weights[1]
        plot_neuron_features(ae_weights, "Autoencoder Hidden Features (20 Random Neurons)",
                             neuron_indices, F_AE_FEATURES_PLOT)

        plot_neuron_features(p1_weights, "Classifier Hidden Features (Matched Neurons)",
                             neuron_indices, F_CLF_FEATURES_PLOT)

    except FileNotFoundError:
        print(f"\nWarning: classifier artifact file '{P1_ARTIFACT_FILE}' not found.")
        print("Skipping the feature-comparison visualizations.")
    except Exception as e:
        print(f"\nError when loading classifier artifacts: {e}")
        print("Skipping the feature-comparison visualizations.")

    ae_weights = ae.weights[1]
    plot_feature_similarities(ae_weights, F_SIM_HEATMAP_PLOT, F_SIM_BAR_PLOT)

    reconstructed_images = ae.reconstruct(X_test)
    plot_sample_outputs(X_test, reconstructed_images, F_RECONSTRUCTION_PLOT)

    print("\n>>> Problem 2 run finished.\n")


# ------------------------------------------------------------------
# Program entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Select which experiment to run:")
    print("  1 - Digit classifier (Problem 1)")
    print("  2 - Autoencoder (Problem 2)")
    print("  3 - Run both in sequence")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        run_problem1()
    elif choice == "2":
        run_problem2()
    elif choice == "3":
        run_problem1()
        run_problem2()
    else:
        print("Unrecognized option; running both experiments.")
        run_problem1()
        run_problem2()
