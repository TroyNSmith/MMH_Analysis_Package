# MMH_Analysis_Package

## Getting Started with Pixi
This project uses [Pixi](https://pixi.sh/latest/) for environment and dependency management. Pixi is a fast, reproducible tool similar to conda, but with better performance, built-in support for scripting and multi-platform environments, and full compatibility with [VS Code](https://code.visualstudio.com/). 

### 1. Prerequisites
#### Git
| Platform | Install Command or Link                                 |
| -------- | ------------------------------------------------------- |
| Windows  | Run through [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) & follow Linux |
| MacOS    | Pre-installed or run `brew install git` (with Homebrew) |
| Linux    | Run `sudo apt install git` (Debian/Ubuntu)              |

Verify installation:
```
git --version
```

#### Pixi
| Platform | Install Command or Link                                 |
| -------- | ------------------------------------------------------- |
| Windows  | Run through [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) & follow Linux |
| Linux    | Run `curl -sSf https://prefix.dev/install.sh`           |

Verify installation:
```
pixi --version
```

Then restart your terminal (or follow the post-install instructions to update your `PATH`).

### 2. Clone This Repository
```
git clone https://github.com/TroyNSmith/MMH_Analysis_Package.git
cd MMH_Analysis_Package
```

### 3. Set Up the Environment
Inside the project folder, **activate** the environment:
```
pixi shell
```
All dependencies and tools will be available automatically in this shell.

### 4. Run Jupyter Notebook for Interactive Docs
Once inside the Pixi environment, run:
```
jupyter lab
```

### 5. Open in VS Code
From the terminal:
```
code .
```

In VS Code:
- Install the **Python** extension.
- Use `ctrl+\` to open the terminal and run `pixi shell`.
- Set the Python interpreter: `Ctrl+Shift+P` &#8594; `Python: Select Interpreter` &#8594; `Pixi env`.

### 6. Push and Pull Changes
#### Pull:
```
git pull origin main
```

#### Commit and Push:
```
git add .
git commit -m "Your message."
git push origin main
```

To do:

1. Change tick number sizes
2. Re-run susceptibility & ISFs
3. Implement changes to non-Gaussian
4. Add a correlation function for radial bins
