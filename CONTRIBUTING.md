# Contributing to Model Distillation Pipeline

Thank you for your interest in contributing to the Model Distillation Pipeline project by Forge1825! We welcome contributions from the community and are grateful for any help you can provide.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/model-distillation-pipeline.git
   cd model-distillation-pipeline
   ```
3. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

1. Follow the setup instructions in [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md)
2. Install development dependencies:
   ```bash
   # Backend
   cd backend
   pip install -r requirements-dev.txt
   
   # Frontend
   cd ../frontend
   npm install --legacy-peer-deps
   ```

## Code Style Guidelines

### Python (Backend)
- Follow PEP 8 style guidelines
- Use type hints where possible
- Maximum line length: 100 characters
- Use docstrings for all functions and classes

### JavaScript/React (Frontend)
- Use ES6+ syntax
- Follow React best practices
- Use functional components with hooks
- Maintain consistent naming conventions

## Making Changes

1. **Write clear commit messages** that explain what you changed and why
2. **Keep commits focused** - one logical change per commit
3. **Test your changes** thoroughly before submitting
4. **Update documentation** if you're changing functionality
5. **Add tests** for new features when possible

## Submitting a Pull Request

1. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Create a Pull Request on GitHub with:
   - Clear title describing the change
   - Detailed description of what was changed and why
   - Reference to any related issues
   - Screenshots for UI changes

3. Ensure all checks pass:
   - Code builds without errors
   - No linting errors
   - Tests pass (when available)

## Code Review Process

- All submissions require review before merging
- Be responsive to feedback and questions
- Make requested changes promptly
- Be patient - reviews may take time

## Reporting Issues

When reporting issues, please include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, etc.)
- Relevant logs or error messages

## Feature Requests

We welcome feature requests! Please:
- Check existing issues first
- Clearly describe the feature
- Explain the use case
- Consider submitting a PR if you can implement it

## Community Guidelines

- Be respectful and professional
- Help others when you can
- Follow our [Code of Conduct](CODE_OF_CONDUCT.md)
- Give credit to Forge1825 in any derivative work

## Development Tips

### Running Tests
```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

### Linting
```bash
# Python
flake8 backend/

# JavaScript
cd frontend
npm run lint
```

## Questions?

If you have questions about contributing, feel free to:
- Open a discussion on GitHub
- Review existing issues and PRs
- Check the documentation

## Attribution

Remember to maintain attribution to Forge1825 as specified in the [LICENSE](LICENSE) file.

Thank you for contributing to Model Distillation Pipeline!