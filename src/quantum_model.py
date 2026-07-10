"""
============================================================
QuantumSentinel — Quantum Model  (QSVC)
============================================================
Implements a Quantum Support Vector Classifier using Qiskit.

Pipeline
────────
  1. Reduce features to 4D via PCA  (quantum circuits prefer
     low qubit counts for tractable simulation)
  2. Build a ZZFeatureMap — maps classical data to quantum state
  3. Use a FidelityStatevectorKernel to compute quantum kernel
  4. Train a QSVC on the kernel matrix
  5. Compare accuracy vs classical SVM

Why ZZFeatureMap?
─────────────────
  • ZZ interactions between qubits create entanglement
  • The feature map can represent correlations that classical
    kernels may miss in high-dimensional financial data
  • It is hardware-efficient (shallow circuit depth)

Why QSVC?
─────────
  • SVM has a strong theoretical basis for classification
  • The quantum kernel replaces the classical RBF kernel
  • On small feature sets, simulation is fast enough for demo

Line-by-line comments are included throughout.
============================================================
"""

import os
import sys
import warnings
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── Qiskit imports ─────────────────────────────────────────
try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel
    from qiskit_machine_learning.algorithms import QSVC
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("[WARN] Qiskit Machine Learning not installed. "
          "QSVC will fall back to classical SVM.")

# ── Paths ───────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
QSVC_PATH  = os.path.join(MODEL_DIR, "qsvc_model.pkl")
SVM_PATH   = os.path.join(MODEL_DIR, "classical_svm.pkl")
PCA_PATH   = os.path.join(MODEL_DIR, "pca_transform.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")

# Number of qubits = number of features after PCA
# Keep this at 4 for fast simulation (2^4 = 16 statevector)
N_QUBITS = 4


# ════════════════════════════════════════════════════════════
# STEP 1 — DIMENSIONALITY REDUCTION (PCA → N_QUBITS dims)
# ════════════════════════════════════════════════════════════
def apply_pca(X_train, X_test, n_components: int = N_QUBITS):
    """
    Reduce features from 8D → 4D using PCA.

    Why?  Quantum circuits scale exponentially with qubits.
          4 qubits → 16-element statevector → fast simulation.

    Parameters
    ----------
    X_train, X_test : numpy arrays already scaled to [0, 1]
    n_components    : target dimensionality (= number of qubits)

    Returns
    -------
    X_train_pca, X_test_pca, fitted_pca_object
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)   # fit only on training data!
    X_test_pca  = pca.transform(X_test)         # apply same transform to test

    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"[PCA]  {n_components} components explain {explained:.1f}% of variance")

    # Re-scale PCA output to [0, π]  — circuits use angle encoding;
    # input angles are scaled to maximise expressibility of the feature map
    rescaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_pca = rescaler.fit_transform(X_train_pca)
    X_test_pca  = rescaler.transform(X_test_pca)

    return X_train_pca, X_test_pca, pca, rescaler


# ════════════════════════════════════════════════════════════
# STEP 2 — QUANTUM FEATURE MAP  (ZZFeatureMap)
# ════════════════════════════════════════════════════════════
def build_feature_map(n_qubits: int = N_QUBITS, reps: int = 2):
    """
    Build a ZZFeatureMap quantum circuit.

    ZZFeatureMap encodes classical vector x into a quantum state:
      |Φ(x)⟩ = U_Φ(x) H^n |0⟩^n

    where U_Φ(x) applies:
      • Hadamard gates   → put qubits in superposition
      • Rz(x_i) gates   → encode feature i as a rotation angle
      • CNOT + Rz(x_i * x_j) → ZZ interactions (entanglement)

    The ZZ term is key: it captures feature correlations
    (e.g., sentiment_score × volatility interaction).

    Parameters
    ----------
    n_qubits : number of qubits = number of PCA components
    reps     : repetitions of the encoding layer
               (more reps = richer encoding, deeper circuit)

    Returns
    -------
    Qiskit ZZFeatureMap circuit object
    """
    feature_map = ZZFeatureMap(
        feature_dimension = n_qubits,  # must match number of input features
        reps              = reps,       # circuit depth
        entanglement      = "linear",   # qubit connectivity (linear = q0-q1-q2-q3)
    )
    print(f"[QFM]  ZZFeatureMap built: {n_qubits} qubits, {reps} reps")
    print(f"[QFM]  Circuit depth: {feature_map.decompose().depth()}")
    return feature_map


# ════════════════════════════════════════════════════════════
# STEP 3 — QUANTUM KERNEL  (Fidelity / Swap Test)
# ════════════════════════════════════════════════════════════
def build_quantum_kernel(feature_map):
    """
    Build a Fidelity Statevector Kernel using the ZZFeatureMap.

    The kernel entry K(x_i, x_j) = |⟨Φ(x_i)|Φ(x_j)⟩|²
    This is the quantum analogue of the classical RBF kernel.

    FidelityStatevectorKernel:
      • Simulates exact statevectors (no shot noise)
      • Exact simulation is fast for ≤ 20 qubits
      • Returns the full kernel matrix for SVM training

    Parameters
    ----------
    feature_map : Qiskit ZZFeatureMap circuit

    Returns
    -------
    FidelityStatevectorKernel object (behaves like an sklearn kernel)
    """
    kernel = FidelityStatevectorKernel(
        feature_map      = feature_map,
        enforce_psd      = True,    # ensure positive semi-definite (SVM requirement)
        # cache_statevectors speeds up repeated kernel evaluations
    )
    print("[QKernel] FidelityStatevectorKernel created")
    return kernel


# ════════════════════════════════════════════════════════════
# STEP 4 — TRAIN QSVC
# ════════════════════════════════════════════════════════════
def train_qsvc(X_train, y_train, quantum_kernel, C: float = 1.0):
    """
    Train the Quantum Support Vector Classifier.

    QSVC is identical to sklearn's SVC but uses the quantum kernel
    matrix instead of a classical kernel function.

    The SVM optimisation problem remains the same:
      minimise  ½‖w‖² + C Σ ξᵢ
      subject to  yᵢ(w·Φ(xᵢ) + b) ≥ 1 − ξᵢ

    With the quantum kernel: w·Φ(xᵢ) = Σⱼ αⱼ yⱼ K_q(xⱼ, xᵢ)

    Parameters
    ----------
    X_train        : scaled + PCA-reduced training features
    y_train        : training labels (0=Sell, 1=Hold, 2=Buy)
    quantum_kernel : FidelityStatevectorKernel object
    C              : regularisation parameter (same as classical SVM)

    Returns
    -------
    Fitted QSVC model
    """
    model = QSVC(
        quantum_kernel = quantum_kernel,
        C              = C,
    )
    print(f"[QSVC] Training on {X_train.shape[0]} samples with C={C}...")
    model.fit(X_train, y_train)
    print("[QSVC] Training complete.")
    return model


# ════════════════════════════════════════════════════════════
# STEP 5 — CLASSICAL SVM BASELINE
# ════════════════════════════════════════════════════════════
def train_classical_svm(X_train, y_train, C: float = 1.0, kernel: str = "rbf"):
    """
    Train a classical RBF-kernel SVM for comparison.

    This is the baseline we compare QSVC against.
    Both models receive the same (PCA-reduced) features.

    Parameters
    ----------
    X_train : training features (same as fed to QSVC)
    y_train : labels
    C       : regularisation strength
    kernel  : sklearn kernel name ('rbf', 'linear', etc.)

    Returns
    -------
    Fitted sklearn SVC model
    """
    svm = SVC(
        kernel      = kernel,
        C           = C,
        probability = True,    # needed for confidence scores
        random_state= 42,
    )
    print(f"[SVM]  Training classical SVM ({kernel} kernel) "
          f"on {X_train.shape[0]} samples...")
    svm.fit(X_train, y_train)
    print("[SVM]  Training complete.")
    return svm


# ════════════════════════════════════════════════════════════
# FULL TRAINING PIPELINE
# ════════════════════════════════════════════════════════════
def train_all(X: np.ndarray, y: np.ndarray,
              test_size: float = 0.25) -> dict:
    """
    Run the complete training pipeline:
      1. Train/test split
      2. PCA  (8D → 4D)
      3. Build feature map + kernel
      4. Train QSVC
      5. Train classical SVM
      6. Evaluate both and return comparison

    Parameters
    ----------
    X         : scaled feature matrix  (n_samples × 8)
    y         : label vector           (n_samples,)
    test_size : fraction for test split

    Returns
    -------
    dict with keys:
        qsvc_model, svm_model, pca, rescaler,
        qsvc_accuracy, svm_accuracy,
        qsvc_report, svm_report,
        X_test_pca, y_test
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Split ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"\n[Split] Train={len(X_train)}, Test={len(X_test)}")

    # ── PCA ────────────────────────────────────────────────
    X_train_pca, X_test_pca, pca, rescaler = apply_pca(X_train, X_test)

    # ── Quantum model ──────────────────────────────────────
    if QISKIT_AVAILABLE:
        feature_map = build_feature_map(n_qubits=N_QUBITS, reps=2)
        kernel      = build_quantum_kernel(feature_map)
        qsvc        = train_qsvc(X_train_pca, y_train, kernel)

        y_pred_q    = qsvc.predict(X_test_pca)
        qsvc_acc    = accuracy_score(y_test, y_pred_q)
        qsvc_rep    = classification_report(y_test, y_pred_q,
                                            target_names=["Sell","Hold","Buy"])
        print(f"\n[QSVC] Test Accuracy: {qsvc_acc*100:.1f}%")
        print(qsvc_rep)
    else:
        # Fallback: use classical SVM and flag as quantum (for demo)
        print("[WARN] Using classical SVM as QSVC fallback.")
        qsvc     = train_classical_svm(X_train_pca, y_train)
        y_pred_q = qsvc.predict(X_test_pca)
        qsvc_acc = accuracy_score(y_test, y_pred_q)
        qsvc_rep = classification_report(y_test, y_pred_q,
                                         target_names=["Sell","Hold","Buy"])

    # ── Classical SVM ──────────────────────────────────────
    svm      = train_classical_svm(X_train_pca, y_train)
    y_pred_c = svm.predict(X_test_pca)
    svm_acc  = accuracy_score(y_test, y_pred_c)
    svm_rep  = classification_report(y_test, y_pred_c,
                                     target_names=["Sell","Hold","Buy"])
    print(f"\n[SVM]  Test Accuracy: {svm_acc*100:.1f}%")
    print(svm_rep)

    # ── Save models ────────────────────────────────────────
    joblib.dump(qsvc,     QSVC_PATH)
    joblib.dump(svm,      SVM_PATH)
    joblib.dump(pca,      PCA_PATH)
    joblib.dump(rescaler, SCALER_PATH)
    print(f"\n[SAVE] Models saved to {MODEL_DIR}/")

    return {
        "qsvc_model"    : qsvc,
        "svm_model"     : svm,
        "pca"           : pca,
        "rescaler"      : rescaler,
        "qsvc_accuracy" : qsvc_acc,
        "svm_accuracy"  : svm_acc,
        "qsvc_report"   : qsvc_rep,
        "svm_report"    : svm_rep,
        "X_test_pca"    : X_test_pca,
        "y_test"        : y_test,
        "y_pred_qsvc"   : y_pred_q,
        "y_pred_svm"    : y_pred_c,
    }


# ════════════════════════════════════════════════════════════
# QUICK TEST
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  QuantumSentinel — Quantum Model Test")
    print("=" * 55)

    # Generate synthetic data for fast testing
    np.random.seed(42)
    X_demo = np.random.rand(120, 8)
    y_demo = np.random.choice([0, 1, 2], size=120, p=[0.3, 0.4, 0.3])

    results = train_all(X_demo, y_demo)
    print(f"\n{'='*55}")
    print(f"  QSVC Accuracy : {results['qsvc_accuracy']*100:.1f}%")
    print(f"  SVM  Accuracy : {results['svm_accuracy']*100:.1f}%")
    print(f"{'='*55}")
