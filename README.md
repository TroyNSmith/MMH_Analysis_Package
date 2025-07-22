# MMH_Analysis_Package

## Getting Started with Pixi
This project uses Pixi for environment and dependency management. Pixi is a fast, reproducible tool similar to conda, but with better performance and built-in support for scripting and multi-platform environments.

### 1. Install Pixi
Run the following in your terminal:
```
curl -sSf https://prefix.dev/install.sh | bash
```

Then restart your terminal (or follow the post-install instructions to update your `PATH`).

Verify installation:
```
pixi --version
```

### 2. Clone This Repository
```
git clone https://github.com/TroyNSmith/MMH_Analysis_Package.git
cd MMH_Analysis_Package
```

### 3. Set Up the Environment
Inside the project folder, activate the environment:
```
pixi shell
```
All dependencies and tools will be available automatically in this shell.

### 4. Run Jupyter Notebook for Interactive Docs
Once inside the Pixi environment, run:
```
jupyter lab
```

To do:

1. Change tick number sizes
2. Re-run susceptibility & ISFs
3. Implement changes to non-Gaussian
4. Add a correlation function for radial bins
