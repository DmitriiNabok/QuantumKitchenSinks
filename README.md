# Quantum Kitchen Sinks

Qiskit-based implementation of the original Quantum Kitchen Sinks (QKS) algorithm by Wilson et al. (2019).
As an extension, the library includes a variant of QKS, where the final circuit measurements are replaced with estimates of the expectation values of single qubits operators.

## References

* [Wilson et al.](https://arxiv.org/abs/1806.08321v2), "Quantum Kitchen Sinks: An Algorithm for Machine Learning on near-Term Quantum Computers", arXiv:1806.08321 (2019)

* [A. Rahimi and B. Recht](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html), "Random Features for Large-Scale Kernel Machines", In Advances in Neural Information Processing Systems; Curran Associates, Inc., 2007; Vol. 20

* [Noori et al.](https://doi.org/10.1103/PhysRevApplied.14.034034) "Analog-Quantum Feature Mapping for Machine-Learning Applications", *Phys. Rev. Applied* **14**, 034034 (2020).


## Installation (with `conda`)

### 0. Clone the repository

```bash
git clone <REPOSITORY>
cd quantum-kitchen-sinks 
```

### 1. Create a virtual environment and activate it

```bash
conda create --name qks python=3
conda activate qks
```

### 2. Install packages (including all requirements)

```bash
pip install -e .
```

### 3. Add the environment to Jupyter Notebooks

```bash
conda install -c ipykernel
python -m ipykernel install --user --name=qks
```


## Usage

Application examples are located in `tutorials`.
