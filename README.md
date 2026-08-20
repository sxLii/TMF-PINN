# TMF-PINN

Physics-informed neural networks for modeling **transient mixed flows in sewer networks**.

## Project branches

The code associated with the published paper is available in the [`paper-version` branch](https://github.com/sxLii/TMF-PINN/tree/paper-version) and [`paper-version-xpinn` branch](https://github.com/sxLii/TMF-PINN/tree/paper-version-xpinn). The [`master` branch](https://github.com/sxLii/TMF-PINN/tree/master) is reserved for the ongoing development of the next-generation implementation.


- 1. Li, S., Tian, W., Yan, H., Zeng, W., Tao, T., & Xin, K. (2024). Modeling transient mixed flows in sewer systems with data fusion via physics-informed machine learning. *Water Research X, 25*, 100266. https://doi.org/10.1016/j.wroa.2024.100266
- 2. Yan, H., Li, S., Tian, W., Wang, J., Li, F., Duan, H., et al. (2025). A hybrid model coupling data and hydraulic transient laws for water distribution systems. Water Resources Research, 61, e2023WR036641. https://doi.org/10.1029/2023WR036641



## Development roadmap

Development of the primary branch is organized into the following five stages:

- [x] **PyTorch version**  
   Original PINN Version implemented in TensorFlow 1.15.x.

- [ ] **Single experimental-case validation**  
   Validate the PyTorch implementation on a single experimental case.

- [ ] **Nodal treatment**  
   Extend the framework to handle nodal variables, junction conditions, and related boundary conditions.

- [ ] **Graph neural network integration**  
   Incorporate graph neural networks to represent sewer-network topology and interactions between pipes and nodes.

- [ ] **Large-scale system application**  
   Scale the framework to larger and more complex sewer-network systems.

The primary development branch is under active development. APIs, training procedures, and data formats may change as each stage is completed.
