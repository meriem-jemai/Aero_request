from setuptools import setup, find_packages

setup(
    name="Aerorequest",
    version="0.1.0",
    description="Custom Gymnasium environment for drone control",
    author="Meriem Jemai",
    packages=find_packages(),
    package_data={"Aerorequest": ["env/resources/*.png"]},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "gymnasium",
        "numpy",
        "pygame-ce",
    ],
    extras_require={"train": ["torch", "stable-baselines3", "matplotlib", "pandas", "pillow"]},
)