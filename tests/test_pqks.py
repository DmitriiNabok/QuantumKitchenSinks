import pytest

import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_allclose

from qiskit.utils import QuantumInstance
from qiskit.circuit import QuantumCircuit, ParameterVector
from qks.ProjectedQuantumKitchenSinks import ProjectedQuantumKitchenSinks

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# Parameters
n_features = 2
n_qubits = 2
n_layers = 1
n_params = 2
seed = 12345

n_episodes = 10
stddev = 3.0

# Feature map circuit setup
theta = ParameterVector("θ", length=n_params)
fm = QuantumCircuit(n_qubits)

j = 0
for r in range(n_layers):
    for i in range(n_qubits):
        fm.ry(theta[j % n_params], i)
        j += 1
    for i in range(n_qubits - 1):
        fm.cx(i, i + 1)
print(fm.draw(fold=120, plot_barriers=False))


def test_init():
    """Test initialization"""
    qks = ProjectedQuantumKitchenSinks(
        n_features,
        fm,
        projection="z",
        n_episodes=n_episodes,
        stddev=stddev,
        sampling="normal",
        seed=seed,
        method="statevector",
    )

    assert isinstance(qks.fm, QuantumCircuit)
    assert qks.n_features == n_features
    assert qks.n_params == n_params
    assert qks.n_episodes == n_episodes
    assert qks.stddev == stddev
    assert qks.sampling == "normal"
    assert qks.projection == "z"
    assert qks.seed == seed
    assert qks.method == "statevector"
    assert isinstance(qks.backend, QuantumInstance)

    assert qks.split == int(n_features / n_params)

    proj_ops = [[("IZ", 1.0)], [("ZI", 1.0)]]
    assert qks.proj_ops == proj_ops


def test_generate_omega_and_beta():
    """QKS random parameter generator"""
    qks = ProjectedQuantumKitchenSinks(
        n_features,
        fm,
        projection="z",
        n_episodes=2,
        stddev=stddev,
        sampling="normal",
        seed=seed,
        method="statevector",
    )
    # print(repr(qks.omega))
    # print(repr(qks.beta))

    omega = np.array(
        [
            [[-0.61412298, -0.0], [0.0, 1.43683001]],
            [[-1.55831615, -0.0], [-0.0, -1.66719091]],
        ]
    )
    assert_allclose(qks.omega, omega)

    beta = np.array([[3.56712156, 3.74191773], [6.06022346, 4.10403274]])
    assert_allclose(qks.beta, beta)


def test_projected_feature_map():
    """Test calculations of the projector expectation values"""
    qks = ProjectedQuantumKitchenSinks(
        n_features,
        fm,
        projection="z",
        n_episodes=1,
        stddev=stddev,
        sampling="normal",
        seed=seed,
        method="statevector",
    )

    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    params = X.dot(qks.omega) + qks.beta

    measures = []
    for param in params:
        for theta in param:
            meas = qks._projected_feature_map(theta)
            measures.append(meas)
    measures = np.array(measures)
    print(repr(measures))

    ref_measures = np.array(
        [
            [0.40337327, 0.11360733],
            [0.85695043, 0.24135425],
            [0.40337327, -0.36840238],
            [0.85695043, -0.78265618],
        ]
    )
    assert_allclose(ref_measures, measures)


def test_embedding():
    """Test the QKS embedding procedure"""
    qks = ProjectedQuantumKitchenSinks(
        n_features,
        fm,
        projection="z",
        n_episodes=1,
        stddev=stddev,
        sampling="normal",
        seed=seed,
        method="statevector",
    )

    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    emb = qks.embedding(X)
    print(repr(emb))

    ref_emb = np.array(
        [
            [0.40337327, 0.11360733],
            [0.85695043, 0.24135425],
            [0.40337327, -0.36840238],
            [0.85695043, -0.78265618],
        ]
    )
    assert_allclose(ref_emb, emb)


def test_qks():
    """Apply QKS to a dataset"""

    X_train = np.array(
        [
            [0.17343808, -0.91095504],
            [-0.22149622, 0.89532844],
            [-0.08353476, -0.30303812],
            [0.247883, 0.19912668],
            [-0.12021119, -0.7508828],
        ]
    )
    y_train = np.array([1, -1, 1, -1, 1])

    X_test = np.array(
        [
            [-0.56751718, 0.51645391],
            [0.14849131, -0.61218433],
            [-0.92375243, 0.40682701],
            [0.00876674, 0.71746302],
            [0.7323017, -0.31563997],
        ]
    )
    y_test = np.array([-1, 1, -1, -1, 1])

    qks = ProjectedQuantumKitchenSinks(
        n_features,
        fm,
        projection="z",
        n_episodes=10,
        stddev=stddev,
        sampling="normal",
        seed=seed,
        method="statevector",
    )

    emb_tr = qks.embedding(X_train)
    clf = LogisticRegression(random_state=seed, max_iter=1000).fit(emb_tr, y_train)

    y_pred = clf.predict(emb_tr)
    acc = metrics.balanced_accuracy_score(y_true=y_train, y_pred=y_pred)
    mcc = metrics.matthews_corrcoef(y_true=y_train, y_pred=y_pred)
    assert_almost_equal(acc, 1.0, decimal=2)
    assert_almost_equal(mcc, 1.0, decimal=2)

    emb_te = qks.embedding(X_test)
    y_pred = clf.predict(emb_te)
    acc = metrics.balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
    mcc = metrics.matthews_corrcoef(y_true=y_test, y_pred=y_pred)
    assert_almost_equal(acc, 1.0, decimal=2)
    assert_almost_equal(mcc, 1.0, decimal=2)


if __name__ == "__main__":

    # test_init()
    # test_generate_omega_and_beta()
    # test_projected_feature_map()
    # test_embedding()
    # test_qks()

    pass
