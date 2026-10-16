# Character controller training

Trains Klartraum's neural character controller (`CharacterController`) and
writes it, with its test fixtures, to `data/character/`.

The controller follows the mixture-of-experts motion networks of Sebastian
Starke and colleagues (mode-adaptive neural networks, Neural State Machine,
Local Motion Phases): a gating network blends the weights of eight expert
networks, which predict the next pose of a character from its current pose and
a trajectory of its root. Klartraum draws the poses as OpenPose pose maps, so
poses are joint positions and velocities rather than rotations; after every
step the limb and face bones are reset to their rest lengths.

The training data is a subset of the **100STYLE dataset** by Ian Mason,
Sebastian Starke and Taku Komura (CC BY 4.0,
https://zenodo.org/records/8127870): twelve everyday locomotion styles, mirrored
copies included, about 525,000 frames at 30 fps.

From this directory:

```bash
uv run python download_100style.py   # ~360 MB of BVH files to build/character_animation/100STYLE
uv run python features.py            # training database build/character_animation/database.npz
uv run python train.py               # 40 epochs, about 15 minutes on an Apple M4
uv run python export_fixtures.py     # initial state and reference rollouts for the tests
```

`download_100style.py` reads the central directory of the 1.5 GB archive with
HTTP range requests and fetches only the selected files. `train.py` exports
`controller.onnx` with normalization built in and checks it against PyTorch;
`--export-only` exports a saved checkpoint again.

| File | Contents |
|---|---|
| `bvh.py` | BVH reader with forward kinematics |
| `features.py` | joint set, feature layout and the training database |
| `train.py` | the network, training and ONNX export |
| `controller.py` | the controller loop that `shaders/character/*.comp` implement on the GPU |
| `export_fixtures.py` | fixtures and plausibility metrics of a scripted rollout |

`controller.py` and the shaders must stay in sync; `tests/test_character_controller.cpp`
compares a GPU rollout with the fixtures written from `controller.py`.
