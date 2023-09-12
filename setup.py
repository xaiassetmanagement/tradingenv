import setuptools
import os
here = os.path.abspath(os.path.dirname(__file__))

# Parse __version__ and __package__. Reference for why it's done that way:
#   https://stackoverflow.com/questions/2058802
with open(os.path.join(here, "tradingenv", "__init__.py"), encoding="utf-8") as f:
    init_file_lines = f.readlines()
for line in init_file_lines:
    if line.startswith(("__package__", "__version__")):
        exec(line.rstrip())

# long_description
with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

# package_data.
package_data_paths = list()
for root, dirs, files in os.walk(os.path.join(here, "tradingenv")):
    for file in files:
        if file.endswith(".csv") or file.endswith(".txt"):
            path = os.path.join(root, file)
            package_data_paths.append(path)

setuptools.setup(
    name=__package__,
    version=__version__,
    description="A trading environment for reinforcement "
    "learning agents, inspired by the OpenAI/gym protocol.",
    long_description=long_description,
    url="https://github.com/xaiassetmanagement/tradingenv",
    author="Federico Fontana",
    author_email="federico.fontana@xai-am.com",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="trading gym",
    packages=setuptools.find_packages(exclude=["docs", "notebooks", "tests"]),
    install_requires=[
        "exchange_calendars",
        "gym==0.21",
        "numpy",
        "pandas",
        "plotly",
        "scikit-learn",
        "statsmodels",
        "tqdm",
    ],
    extras_require={
        "extra": [
            "yfinance",
            "pytest",
            "pytest-mock",
            "pytest-cov",
            "openpyxl",
            "jupyterlab",
            "nbconvert",  # https://stackoverflow.com/questions/65376052
            "Sphinx",
        ],
    },
    package_data={"tradingenv": package_data_paths},
)
