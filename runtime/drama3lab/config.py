from sglang_omni.config.schema import StageConfig
from sglang_omni.models.fishaudio_drama3.config import Drama3PipelineConfig


class CachedDrama3PipelineConfig(Drama3PipelineConfig):
    stages: list[StageConfig] = [
        stage.model_copy(deep=True, update={
            'factory_path': 'drama3_runtime.create_preprocessing_executor'}
        ) if stage.name == 'preprocessing' else stage.model_copy(deep=True)
        for stage in Drama3PipelineConfig.model_fields['stages'].default
    ]


EntryClass = CachedDrama3PipelineConfig


class StreamingCachedDrama3PipelineConfig(CachedDrama3PipelineConfig):
    stages: list[StageConfig] = [
        stage.model_copy(deep=True, update={
            'factory_path': 'drama3_streaming.create_streaming_vocoder_executor'}
        ) if stage.name == 'vocoder' else stage.model_copy(deep=True)
        for stage in CachedDrama3PipelineConfig.model_fields['stages'].default
    ]
