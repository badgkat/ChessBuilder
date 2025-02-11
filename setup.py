from setuptools import setup, find_packages

setup(
    name="chess-builder",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pygame>=2.6.1",
        "pyperclip>=1.9.0",
    ],
    python_requires=">=3.7",
    extras_require={
        "dev": [
            "pytest>=8.3.4",
        ],
    }
)
