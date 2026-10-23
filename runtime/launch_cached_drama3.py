"""Register our pipeline topology, then use the unchanged upstream CLI."""
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.cli import app
import os
from drama3lab.config import CachedDrama3PipelineConfig, StreamingCachedDrama3PipelineConfig


if __name__ == '__main__':
    config = (StreamingCachedDrama3PipelineConfig if os.environ.get('DRAMA3_STREAMING_ONLY') == '1'
              else CachedDrama3PipelineConfig)
    PIPELINE_CONFIG_REGISTRY.configs[config.architecture] = config
    app()
