# End-to-end reinforcement learning for autonomous longitudinal control using advantage actor critic with temporal context

This is the repo for paper End-to-end reinforcement learning for autonomous longitudinal control using advantage actor critic with temporal context. 
Trains an A2C policy to control an autonomous vehicle in a highway, vehicle following setting.


## Installation

Clone the repo

```bash
git clone https://github.com/sampo-kuutti/e2e-rl-longitudinal-control
```

install requirements:
```bash
pip install -r requirements.txt
```

## Training the policy

For training the model, run `train_a2c.py`.

## Citing the Repo

If you find the code useful in your research or wish to cite it, please use the following BibTeX entry.

```text
@inproceedings{kuutti2019end,
  title={End-to-end reinforcement learning for autonomous longitudinal control using advantage actor critic with temporal context},
  author={Kuutti, Sampo and Bowden, Richard and Joshi, Harita and de Temple, Robert and Fallah, Saber},
  booktitle={2019 IEEE Intelligent Transportation Systems Conference (ITSC)},
  pages={2456--2462},
  year={2019},
  organization={IEEE}
}
```