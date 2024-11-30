# Contributing Guidelines

## Development Workflow

1. **Setup Development Environment**
```bash
make install  # Installs package with dev dependencies
```

2. **Code Quality**
- Format code before committing:
```bash
make format  # Runs black and isort
```
- Run linting checks:
```bash
make lint    # Runs flake8, black --check, and isort --check
```

3. **Testing**
```bash
# Run all tests with coverage
make test

# Run specific test suites
make test-pipeline   # Run end-to-end pipeline tests
```

4. **Model Development**
- Follow the existing pattern in `src/models/`
- Add appropriate tests in `tests/models/`
- Validate model performance:
```bash
make validate  # Runs validation suite
```

5. **Cleanup**
- Use cleanup scripts when testing:
```bash
./scripts/cleanup_all.sh  # Clean all generated files
```

## Pull Request Process
1. Create a feature branch
2. Add tests for new functionality
3. Ensure all tests pass
4. Update documentation
5. Submit PR with clear description

## Code Style
- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions/classes
- Keep functions focused and modular

## Documentation
- Update README.md for user-facing changes
- Document complex algorithms
- Include example usage
- Update configuration files
- Update validation criteria when changed

## Directory Structure
Follow the established directory structure:
```
src/          # Source code
tests/        # Test suite
configs/      # Configuration files
models/       # Saved models
```
