# Lint as: python3
""" HuggingFace/Evaluate is an open-source library for evaluating machine learning benchmarks.

Note:

   VERSION needs to be formatted following the MAJOR.MINOR.PATCH convention
   (we need to follow this convention to be able to retrieve versioned scripts)

Simple check list for release from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py

To create the package for pypi.

1. Change the version in __init__.py, setup.py as well as docs/source/conf.py.

2. Commit these changes with the message: "Release: VERSION"

3. Add a tag in git to mark the release: "git tag VERSION -m'Adds tag VERSION for pypi' "
   Push the tag to git: git push --tags origin master

4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.

5. Check that everything looks correct by uploading the package to the pypi test server:

   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/

   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi datasets

6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi

7. Fill release notes in the tag in github once everything is looking hunky-dory.

8. Update the documentation commit in .circleci/deploy.sh for the accurate documentation to be displayed
   Update the version mapping in docs/source/_static/js/custom.js with utils/release.py,
   and set version to X.X.X+1.dev0 (e.g. 1.8.0 -> 1.8.1.dev0) in setup.py and __init__.py

"""

from pathlib import Path

from setuptools import find_packages, setup

DOCLINES = __doc__.split("\n")


REQUIRED_PKGS = ["datasets==1.11.0"]

QUALITY_REQUIRE = ["black", "flake8", "isort", "pyyaml>=5.3.1"]

TESTS_REQUIRE = ["pytest", "pytest-cov"]

EXTRAS_REQUIRE = {"quality": QUALITY_REQUIRE, "tests": TESTS_REQUIRE}


def combine_requirements(base_keys):
    return list(set(k for v in base_keys for k in EXTRAS_REQUIRE[v]))


EXTRAS_REQUIRE["dev"] = combine_requirements([k for k in EXTRAS_REQUIRE])

benchmark_dependencies = list(Path("benchmarks/").glob("**/requirements.txt"))
for benchmark in benchmark_dependencies:
    with open(benchmark, "r") as f:
        deps = f.read().splitlines()
        EXTRAS_REQUIRE[benchmark.parent.name] = deps

setup(
    name="evaluate",
    version="0.0.1",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    author="HuggingFace Inc.",
    author_email="lewis@huggingface.co",
    url="https://github.com/huggingface/evaluate",
    download_url="https://github.com/huggingface/evaluate/tags",
    package_dir={"": "src"},
    packages=find_packages("src"),
    license="Apache 2.0",
    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine learning benchmarks evaluation metrics",
    zip_safe=False,  # Required for mypy to find the py.typed file
)
