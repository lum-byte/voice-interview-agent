import asyncio
from typing import Dict, Any

from app.orchestration.voice_graph import voice_graph


class VoicePipeline:
    """
    Object-oriented wrapper around the voice orchestration graph.

    Handles:
    - running the pipeline
    - formatting outputs
    - future extensions (auth, rate limits, configs, tracing)
    """

    def __init__(self):
        self._graph = voice_graph

    async def run(self, audio_path: str) -> Dict[str, Any]:
        """
        Runs full voice → voice pipeline.

        Flow:
        audio → STT → LLM → TTS → output audio
        """

        result = await self._graph.run({"audio_path": audio_path, "mode": "api"})

        return {
            "transcript": result.get("transcript"),
            "llm_response": result.get("llm_response"),
            "audio_output": result.get("audio_output"),
        }


# Local test runner
if __name__ == "__main__":
    pipeline = VoicePipeline()
    test_audio = "audio/audio_IN/stt_test.wav"

    output = asyncio.run(pipeline.run(test_audio))

    print("\n--- PIPELINE RESULT ---")
    print("Transcript:", output["transcript"])
    print("LLM:", output["llm_response"])
    print("Output Audio:", output["audio_output"])
