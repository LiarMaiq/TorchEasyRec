# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Predictor for loading a scripted model once and performing multiple predictions."""

import os
from typing import Dict, Optional

import pyarrow as pa
import torch

from tzrec.acc import aot_utils
from tzrec.acc import utils as acc_utils
from tzrec.main import _create_features
from tzrec.utils import config_util
from tzrec.utils.logging_util import logger


class Predictor:
    """Predictor for loading a scripted model once and performing multiple predictions.

    This class separates model loading/initialization from the prediction step,
    avoiding repeated model loading overhead when making multiple predictions.

    Example:
        predictor = Predictor("/path/to/scripted_model")
        predictions1 = predictor.predict(raw_data1)
        predictions2 = predictor.predict(raw_data2)
    """

    def __init__(
        self, scripted_model_path: str, device: Optional[torch.device] = None
    ) -> None:
        """Initialize the predictor by loading model, features and data parser.

        Args:
            scripted_model_path (str): path to the exported scripted model directory.
            device (torch.device, optional): inference device. Defaults to cpu.
        """
        from tzrec.datasets.data_parser import DataParser, Mode

        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self._device = device
        self._scripted_model_path = scripted_model_path

        # Load pipeline config to initialize features and data parser
        self._pipeline_config = config_util.load_pipeline_config(
            os.path.join(scripted_model_path, "pipeline.config"), allow_unknown_field=True
        )
        train_config = self._pipeline_config.train_config
        acc_utils.allow_tf32(train_config)

        # Build features from config
        self._features = _create_features(
            list(self._pipeline_config.feature_configs),
            self._pipeline_config.data_config,
        )

        # Create data parser for feature engineering
        self._data_parser = DataParser(
            self._features,
            labels=[],
            sample_weights=[],
            mode=Mode.PREDICT,
            fg_threads=self._pipeline_config.data_config.fg_threads,
            force_base_data_group=self._pipeline_config.data_config.force_base_data_group,
            sampler_type=None,
        )

        # Determine model type and load appropriately
        self._is_trt: bool = acc_utils.is_trt_predict(scripted_model_path)
        self._is_aot: bool = acc_utils.is_aot_predict(scripted_model_path)
        self._is_input_tile: bool = acc_utils.is_input_tile_predict(scripted_model_path)

        if self._is_trt:
            max_batch_size = acc_utils.get_max_export_batch_size()
            logger.info("trt predict mode, max_batch_size: %s", max_batch_size)

        if self._is_aot:
            self._model: aot_utils.CombinedModelWrapper = aot_utils.load_model_aot(
                scripted_model_path, device=device
            )
        else:
            # disable jit compile， as it compile too slow now.
            if "PYTORCH_TENSOREXPR_FALLBACK" not in os.environ:
                os.environ["PYTORCH_TENSOREXPR_FALLBACK"] = "2"
            self._model: torch.jit.ScriptModule = torch.jit.load(
                os.path.join(scripted_model_path, "scripted_model.pt"),
                map_location=device,
            )
            self._model.eval()

    def predict(
        self,
        raw_data: Dict[str, "pa.Array"],
    ) -> Dict[str, torch.Tensor]:
        """Perform prediction on raw input data.

        Args:
            raw_data: Dictionary mapping column names to PyArrow Arrays.
                     This is the exact format required by DataParser.
                     Example format:
                     {
                         "user_id": pa.array([1, 2, 3]),
                         "item_id": pa.array([100, 200, 300]),
                         "score": pa.array([0.5, 0.8, 0.9], type=pa.float32()),
                     }

        Returns:
            predictions (dict): a dict of predicted result tensors.

        Raises:
            ValueError: If raw_data format is incorrect.

        Note:
            The caller (e.g., web service layer) is responsible for converting
            JSON/other formats to pyarrow.Arrays before calling this method.
        """
        # Validate input format
        if not isinstance(raw_data, dict):
            raise ValueError(
                f"raw_data must be a dict mapping column names to pyarrow.Arrays, "
                f"got {type(raw_data)}"
            )
        
        for key, value in raw_data.items():
            if not isinstance(value, pa.Array):
                raise ValueError(
                    f"Value for key '{key}' must be a pyarrow.Array, "
                    f"got {type(value)}. "
                    f"Please convert your data to pyarrow.Arrays before calling predict()."
                )
        
        # Parse features through DataParser
        parsed_features = self._data_parser.parse(raw_data)

        if self._is_input_tile:
            parsed_features["batch_size"] = torch.tensor(1, dtype=torch.int64)

        with torch.no_grad():
            predictions = self._model(parsed_features, self._device)
            if self._device.type == "cuda":
                predictions = {k: v.to("cpu") for k, v in predictions.items()}

        return predictions
