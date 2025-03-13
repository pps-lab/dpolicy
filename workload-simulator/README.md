<a name="readme-top"></a>
# Workload Generator

<!-- ABOUT THE PROJECT -->
## About The Project

This project extends Cohere's configurable [DP workload generator](https://github.com/pps-lab/cohere/tree/main/workload-simulator), leveraging its existing support for diverse request types and workload characteristics.
We have extended the simulator to handle requests with varying contexts, enable attribute tagging, and incorporate the option for time-based privacy units.
Moreover, DPolicy functionalities are integrated directly: policies are configured in `workload-simulator/workload_simulator/policy/budget_policy.py`, and the rule set optimization is implemented within `workload-simulator/workload_simulator/policy/rules.py`.


### Built With

* [![Python][python-shield]][python-url]
* [![Poetry][poetry-shield]][poetry-url]
* [![AutoDP][autodp-shield]][autodp-url]
* [![SimPy][simpy-shield]][simpy-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

* [Python Poetry](https://python-poetry.org/)
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

* [Make](https://www.gnu.org/software/make/)

* Local clone of the repository (with submodules)
    ```sh
    git clone --recurse-submodules git@github.com:pps-lab/dpolicy.git
    ```


### Installation


To re-create the workloads for the three scenarios from the evaluation section in the paper:
```sh
# takes ~6 mins
make create-workloads
```

This will create the following directories with the workloads and their variations:

    ```sh
    doe-suite-results-super-reproduce/
    ├─ ...
    └─ workloads
        ├─ 20-1w-12w-morecat
        ├─ 20-1w-12w-relaxation
        └─ 20-1w-12w-time
    ```


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

In [workload-simulator/workload_simulator/main.py](workload_simulator/main.py), we provide the workload configurations used in DPolicy's evaluation.
The function `default_scenario()` contains Cohere's evaluation scenario, while `mixed_workload(..)` contains the extended version of Cohere's mixed workload used in the evaluation.


<!-- MARKDOWN LINKS & IMAGES -->

[python-shield]: https://img.shields.io/badge/python-grey?style=for-the-badge&logo=python
[python-url]: https://www.python.org/

[poetry-shield]: https://img.shields.io/badge/poetry-grey?style=for-the-badge&logo=poetry
[poetry-url]: https://python-poetry.org/


[autodp-shield]: https://img.shields.io/badge/autodp-grey?style=for-the-badge&logo=github
[autodp-url]: https://github.com/yuxiangw/autodp


[simpy-shield]: https://img.shields.io/badge/simpy-grey?style=for-the-badge&logo=pypi
[simpy-url]: https://simpy.readthedocs.io/en/latest/
