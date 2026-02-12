from setuptools import setup, find_packages

setup(
    name="wrds-data",
    version="0.1.0",
    description="WRDS financial data library with academic-quality corrections for ML pipelines",
    author="Joseph Denman",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyarrow>=12.0.0",
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
        "wrds>=3.2.0",
        "loguru>=0.7.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
)
