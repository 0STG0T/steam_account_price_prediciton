from setuptools import setup, find_packages

setup(
    name="ml_price_predictor",
    version="1.0.0",
    packages=find_packages() + ['scripts'],
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "catboost>=1.2.0",
        "onnxruntime>=1.15.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0.0",
        "plotly>=4.14.0",
        "jsonschema>=4.17.0",
        "dash>=2.9.0",
        "dash-bootstrap-components>=1.4.0",
        "kaleido>=0.2.1",  # Required for plotly static image export
        "statsmodels>=0.14.0",  # Required for LOWESS smoothing
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ]
    },
    author="Devin AI",
    description="ML-based price prediction system for gaming accounts",
    python_requires=">=3.8",
)
