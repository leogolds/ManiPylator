# Contributing to ManiPylator

Thank you for your interest in contributing to ManiPylator! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Contributing Guidelines](#contributing-guidelines)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Documentation](#documentation)
- [Code Style](#code-style)
- [Release Process](#release-process)

## Code of Conduct

This project is committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Docker
- [Dagger CLI](https://docs.dagger.io/quickstart/cli) (v0.18.12)
- Git

### Local Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ManiPylator.git
   cd ManiPylator
   ```

2. **Set up Dagger engine (recommended)**
   ```bash
   cd containers/dagger-engine
   ./reload-dagger-engine.sh
   export _EXPERIMENTAL_DAGGER_RUNNER_HOST=docker-container://dagger-engine-custom
   ```

3. **Install development dependencies**
   ```bash
   # Using pip
   pip install -e ".[dev]"
   
   # Or using uv (recommended)
   uv sync --dev
   ```

4. **Start the development environment**
   ```bash
   # Using Dagger (recommended)
   dagger call controller up --source=. --virtual-mode=true
   
   # Or using Docker Compose
   docker compose up -d
   ```

### Alternative Setup with Docker Compose

```bash
# Start Jupyter Lab environment
docker compose up -d

# Allow X11 forwarding for Genesis window
xhost +local:root
```

Access Jupyter Lab at [http://localhost:8888/lab](http://localhost:8888/lab)

## Project Structure

```
ManiPylator/
├── manipylator/          # Main Python library
│   ├── __init__.py       # Package initialization
│   ├── app.py           # Main application logic
│   ├── base.py          # Base classes and utilities
│   ├── comms.py         # Communication modules
│   ├── utils.py         # Utility functions
│   └── sdk/             # SDK components
├── controller/           # Klipper configuration and macros
├── robots/              # URDF models and robot configurations
├── containers/          # Docker containers and configurations
│   ├── dagger-engine/   # Dagger engine configuration
│   ├── controller/      # Controller container
│   └── lab/            # Lab environment container
├── *.ipynb              # Jupyter notebooks with examples
└── dagger.json          # Dagger module configuration
```

## Contributing Guidelines

### Types of Contributions

We welcome contributions in the following areas:

- **Bug fixes**: Report and fix bugs
- **New features**: Add new functionality
- **Documentation**: Improve docs, tutorials, and examples
- **Testing**: Add tests and improve test coverage
- **Performance**: Optimize code and improve efficiency
- **Examples**: Create new Jupyter notebooks and tutorials

### Before You Start

1. **Check existing issues**: Search for existing issues or discussions
2. **Create an issue**: For significant changes, create an issue first
3. **Discuss**: Engage with the community about your proposed changes

### Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Run tests** locally
7. **Commit your changes** with clear commit messages
8. **Push to your fork**
9. **Create a Pull Request**

### Commit Message Guidelines

Use clear, descriptive commit messages:

```
feat: add inverse kinematics solver for 6DOF manipulator
fix: resolve MQTT connection timeout issue
docs: update README with new installation instructions
test: add unit tests for trajectory planning module
style: format code according to black standards
```

## Development Workflow

### Using Dagger (Recommended)

```bash
# Build containers
dagger call build-lab --source=.
dagger call build-controller --source=.

# Run tests
dagger call controller test --source=.

# Run simulations
dagger call controller simulate --source=. --notebook=your-notebook.ipynb

# Publish containers
dagger call publish-lab --source=. --tag=latest
```

### Local Development

```bash
# Install in development mode
pip install -e .

# Run tests
pytest

# Run with coverage
pytest --cov=manipylator

# Format code
black manipylator/
isort manipylator/

# Lint code
flake8 manipylator/
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=manipylator --cov-report=html

# Run specific test file
pytest tests/test_kinematics.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Place tests in `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies
- Aim for high test coverage

Example test structure:
```python
def test_forward_kinematics():
    """Test forward kinematics calculation."""
    robot = Robot()
    joint_angles = [0, 0, 0, 0, 0, 0]
    pose = robot.forward_kinematics(joint_angles)
    assert pose.shape == (4, 4)
```

## Documentation

### Code Documentation

- Use docstrings for all public functions and classes
- Follow Google or NumPy docstring format
- Include type hints for function parameters and return values

Example:
```python
def calculate_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
    """Calculate the Jacobian matrix for given joint angles.
    
    Args:
        joint_angles: Array of joint angles in radians
        
    Returns:
        Jacobian matrix of shape (6, n_joints)
        
    Raises:
        ValueError: If joint_angles has incorrect shape
    """
```

### Notebook Documentation

- Keep notebooks up to date with code changes
- Include clear explanations and comments
- Use markdown cells for documentation
- Test notebooks regularly

## Code Style

### Python Code

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Maximum line length: 88 characters (Black default)

### Configuration Files

- Use consistent indentation (spaces, not tabs)
- Group related settings together
- Add comments for complex configurations

### Pre-commit Hooks

Set up pre-commit hooks for automatic formatting:

```bash
pip install pre-commit
pre-commit install
```

## Release Process

### Version Management

- Use semantic versioning (MAJOR.MINOR.PATCH)
- Update version in `pyproject.toml`
- Create release notes for significant changes

### Release Checklist

- [ ] Update version number
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release tag
- [ ] Publish to PyPI (if applicable)

## Getting Help

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and general discussion
- **Documentation**: Check the README and Jupyter notebooks for examples

## License

By contributing to ManiPylator, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Thank you to all contributors who have helped make ManiPylator what it is today! 