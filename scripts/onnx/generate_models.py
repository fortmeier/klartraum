"""Generate the deterministic ONNX fixtures used by Klartraum's tests."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from datasets import load_dataset
from onnx import helper, shape_inference
from torch.utils.data import DataLoader
from tqdm import tqdm


SEED = 20250925
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data"
ONNX_DIR = DATA_DIR / "onnx"
TMP_DIR = ONNX_DIR / "tmp"

RAW_ENCODER = TMP_DIR / "simple_encoder.onnx"
RAW_DECODER = TMP_DIR / "simple_decoder.onnx"
ENCODER_WITH_SHAPES = TMP_DIR / "simple_encoder_with_value_info.onnx"
DECODER_WITH_SHAPES = TMP_DIR / "simple_decoder_with_value_info.onnx"
ENCODER_WITH_PARAMS = ONNX_DIR / "simple_encoder.onnx"
DECODER_WITH_PARAMS = ONNX_DIR / "simple_decoder.onnx"
FROZEN_ENCODER = ONNX_DIR / "simple_encoder_with_onnx_frozen_intermediates.onnx"
FROZEN_DECODER = ONNX_DIR / "simple_decoder_with_onnx_frozen_intermediates.onnx"

PUBLISHED_ARTIFACTS = (ENCODER_WITH_PARAMS, DECODER_WITH_PARAMS, FROZEN_ENCODER, FROZEN_DECODER)
INTERMEDIATE_ARTIFACTS = (
    RAW_ENCODER,
    RAW_DECODER,
    ENCODER_WITH_SHAPES,
    DECODER_WITH_SHAPES,
)


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.relu(self.conv3(x))


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 3):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        return self.relu(self.deconv3(x))


def configure_determinism() -> torch.device:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_training_loader(transform) -> DataLoader:
    sample_count = int(os.environ.get("KLARTRAUM_ONNX_TRAIN_SAMPLES", "0"))
    dataset = load_dataset("bitmind/caltech-101", split="train")
    if sample_count > 0:
        dataset = dataset.select(range(min(sample_count, len(dataset))))

    def prepare_batch(examples):
        return {"image": [transform(image.convert("RGB")) for image in examples["image"]]}

    dataset.set_transform(prepare_batch)
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


def train_autoencoder(transform, device: torch.device) -> tuple[Encoder, Decoder]:
    print(f"Training device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    autoencoder = nn.Sequential(encoder, decoder)
    autoencoder.train()

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    loss_function = nn.MSELoss()
    for batch in tqdm(make_training_loader(transform), desc="Training fixture model"):
        images = batch["image"].to(device)
        optimizer.zero_grad()
        reconstruction = autoencoder(images)
        loss = loss_function(reconstruction, images)
        loss.backward()
        optimizer.step()

    autoencoder.eval().cpu()
    return encoder, decoder


def export_model(model, model_input: torch.Tensor, path: Path) -> None:
    torch.onnx.export(
        model,
        model_input,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )


def add_initializers_to_value_info(model: onnx.ModelProto) -> onnx.ModelProto:
    existing_names = {value.name for value in model.graph.value_info}
    existing_names.update(value.name for value in model.graph.input)
    for tensor in model.graph.initializer:
        if tensor.name not in existing_names:
            model.graph.value_info.append(
                helper.make_tensor_value_info(
                    tensor.name,
                    tensor.data_type,
                    list(tensor.dims),
                )
            )
    return model


def add_shape_and_parameter_info(source: Path, shaped: Path, parameterized: Path) -> None:
    model = shape_inference.infer_shapes(onnx.load(source))
    onnx.save(model, shaped)
    onnx.save(add_initializers_to_value_info(model), parameterized)


def value_info_by_name(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    return {value.name: value for value in values}


def freeze_intermediate_values(
    model_path: Path,
    input_data: torch.Tensor,
    output_path: Path,
) -> None:
    model = onnx.load(model_path)
    reference_names = [model.graph.input[0].name]
    reference_names.extend(output for node in model.graph.node for output in node.output)
    reference_names = list(dict.fromkeys(reference_names))

    capture_model = onnx.load(model_path)
    known_values = value_info_by_name(capture_model)
    existing_outputs = {value.name for value in capture_model.graph.output}
    for name in reference_names:
        if name not in existing_outputs:
            if name not in known_values:
                raise RuntimeError(f"No inferred type/shape information for {name}")
            capture_model.graph.output.append(known_values[name])

    capture_path = model_path.with_name(f"{model_path.stem}_capture.onnx")
    onnx.save(capture_model, capture_path)
    try:
        session = ort.InferenceSession(capture_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        input_array = input_data.detach().cpu().numpy()
        reference_values = session.run(reference_names, {input_name: input_array})
    finally:
        capture_path.unlink(missing_ok=True)

    for name, value in zip(reference_names, reference_values):
        value = np.asarray(value)
        if value.dtype != np.float32:
            raise RuntimeError(f"Expected float32 reference tensor {name}, got {value.dtype}")
        capture_model.graph.initializer.append(
            helper.make_tensor(
                name=name,
                data_type=onnx.TensorProto.FLOAT,
                dims=value.shape,
                vals=value.flatten().tolist(),
            )
        )
    onnx.save(capture_model, output_path)

    initializer_names = {tensor.name for tensor in capture_model.graph.initializer}
    missing = set(reference_names) - initializer_names
    if missing:
        raise RuntimeError(f"Frozen model is missing reference tensors: {sorted(missing)}")


def validate_artifacts() -> tuple[Path, ...]:
    artifacts = PUBLISHED_ARTIFACTS + INTERMEDIATE_ARTIFACTS
    missing = [path for path in artifacts if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing or empty ONNX artifacts: {missing}")

    for path in artifacts:
        model = onnx.load(path)
        if path not in (FROZEN_ENCODER, FROZEN_DECODER):
            onnx.checker.check_model(model)
        print(f"Generated {path.relative_to(REPO_ROOT)} ({path.stat().st_size:,} bytes)")

    for path in (FROZEN_ENCODER, FROZEN_DECODER):
        model = onnx.load(path)
        reference_names = [model.graph.input[0].name]
        reference_names.extend(output for node in model.graph.node for output in node.output)
        initializers = {tensor.name: tensor for tensor in model.graph.initializer}
        for name in reference_names:
            tensor = initializers[name]
            expected_elements = int(np.prod(tensor.dims))
            if tensor.raw_data or len(tensor.float_data) != expected_elements:
                raise RuntimeError(
                    f"Reference tensor {name} in {path.name} must use float_data "
                    f"with {expected_elements} elements"
                )
    return artifacts


def generate_models() -> tuple[Path, ...]:
    device = configure_determinism()
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for stale_model in TMP_DIR.glob("*.onnx"):
        stale_model.unlink()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((128, 128), antialias=True),
        ]
    )
    encoder, decoder = train_autoencoder(transform, device)

    from PIL import Image

    image = Image.open(DATA_DIR / "lantern.jpg").convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        latent = encoder(image_tensor)

    export_model(encoder, image_tensor, RAW_ENCODER)
    export_model(decoder, latent, RAW_DECODER)
    add_shape_and_parameter_info(RAW_ENCODER, ENCODER_WITH_SHAPES, ENCODER_WITH_PARAMS)
    add_shape_and_parameter_info(RAW_DECODER, DECODER_WITH_SHAPES, DECODER_WITH_PARAMS)
    freeze_intermediate_values(ENCODER_WITH_PARAMS, image_tensor, FROZEN_ENCODER)
    freeze_intermediate_values(DECODER_WITH_PARAMS, latent, FROZEN_DECODER)
    return validate_artifacts()


if __name__ == "__main__":
    generate_models()
