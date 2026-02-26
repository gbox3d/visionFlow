__all__ = ["GenerateTranscriber", "PipelineTranscriber", "CT2Transcriber"]


def __getattr__(name: str):
    if name == "CT2Transcriber":
        from voiceFlow.vendors.miso_stt.backends.ct2 import CT2Transcriber

        return CT2Transcriber
    if name == "GenerateTranscriber":
        from voiceFlow.vendors.miso_stt.backends.hf_generate import GenerateTranscriber

        return GenerateTranscriber
    if name == "PipelineTranscriber":
        from voiceFlow.vendors.miso_stt.backends.hf_pipeline import PipelineTranscriber

        return PipelineTranscriber
    raise AttributeError(name)

