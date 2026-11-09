from dataclasses import dataclass
import io
from pathlib import Path
import re
import wave

from aiohttp import ClientError, ClientSession, ClientTimeout
import msgpack

from astrbot.api import logger

from .config import PluginConfig


FISH_EMOTIONS = frozenset({
    "soft tone", "slightly shy", "slightly angry", "slightly surprised",
    "warm and happy", "concerned", "relieved", "slightly sad",
})


def normalize_pause_tags(text: str, enabled: bool, marker: str = "<break time=100>") -> str:
    remaining = 1 if enabled else 0

    def replace(match):
        nonlocal remaining
        if remaining and text[:match.start()].strip() and text[match.end():].strip():
            remaining -= 1
            return marker
        return "、"

    return re.sub(r"[、,，…]*(?:<break\b[^>]*>|\[break\])[、,，…]*", replace, text, flags=re.IGNORECASE)


def normalize_fish_text(text: str, keep_tags: bool = True) -> str:
    if not keep_tags:  # Fish 上存好的声音模型不加任何标记（用户试听过，不带更自然）；停顿改成逗号
        text = re.sub(r"\[(?!break\])[A-Za-z][A-Za-z -]*\]", "", text)
        return re.sub(r"([。、！？])、", lambda m: m.group(1), normalize_pause_tags(text, False)).strip()
    emotion = None

    def replace(match):
        nonlocal emotion
        cue = match.group(1).strip().lower()
        if cue == "break":
            return "[break]"
        if cue in FISH_EMOTIONS and emotion in (None, "soft tone"):
            emotion = cue
        return ""

    text = re.sub(r"\[([A-Za-z][A-Za-z -]*)\]", replace, text)
    text = normalize_pause_tags(text, True, marker="[break]").strip()
    if not text:
        return ""
    return f"[{emotion}]{text}" if emotion else text  # 没写标记就用声音本来的语调，不再默认补 [soft tone]


def normalize_wav(data: bytes) -> bytes:
    # Streaming WAV headers can declare an unknown length; QQ needs the actual size.
    with wave.open(io.BytesIO(data), "rb") as source:
        params = source.getparams()
        pcm = source.readframes(params.nframes)
    if not pcm or len(pcm) % (params.nchannels * params.sampwidth):
        raise ValueError("Fish returned empty or incomplete WAV audio")
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setparams(params._replace(nframes=0))
        target.writeframes(pcm)
    return output.getvalue()


@dataclass
class GSVRequestResult:
    ok: bool
    data: bytes | None = None
    error: str = ""
    text: str = ""
    file_path: str = ""

    @property
    def size(self) -> int:
        """音频数据大小（字节）"""
        return len(self.data) if self.data else 0

    @property
    def is_empty(self) -> bool:
        """是否无数据"""
        return self.size == 0

    def __bool__(self) -> bool:
        return self.ok and not self.is_empty



class GSVApiClient:
    """
    API 层（HTTP 通信）
    """

    def __init__(self, config: PluginConfig):
        self.cfg = config.client
        self.base_url = self.cfg.base_url.rstrip("/")
        self.gpt_url = f"{self.base_url}/set_gpt_weights"
        self.sovits_url = f"{self.base_url}/set_sovits_weights"
        self.control_url = f"{self.base_url}/control"
        self.tts_url = f"{self.base_url}/tts"

        self.session = ClientSession(timeout=ClientTimeout(total=self.cfg.timeout))

    async def close(self):
        if self.session:
            await self.session.close()

    async def _request(
        self,
        url: str,
        *,
        params: dict | None = None,
        payload: dict | None = None,
        packed: bool = False,
    ) -> GSVRequestResult:
        request_text = str((payload or params or {}).get("text", ""))
        if params:
            params = {
                k: str(v).lower() if isinstance(v, bool) else v
                for k, v in params.items()
            }

        try:
            headers = {"Authorization": f"Bearer {self.cfg.api_key}"} if payload is not None else None
            if packed:
                headers.update({"model": self.cfg.model, "Content-Type": "application/msgpack"})
            async with self.session.request(
                "POST" if payload is not None else "GET", url,
                params=params, json=None if packed else payload, headers=headers,
                data=msgpack.packb(payload, use_bin_type=True) if packed else None,
            ) as resp:
                if resp.status != 200:
                    detail = await resp.text()
                    if payload is not None and self.cfg.api_key:
                        detail = detail.replace(self.cfg.api_key, "[redacted]")
                    return GSVRequestResult(
                        ok=False,
                        error=f"HTTP {resp.status}: {detail}",
                        text=request_text,
                    )

                if payload is None or packed:
                    audio = await resp.read()
                    if packed and (audio[:4] != b"RIFF" or audio[8:12] != b"WAVE"):
                        return GSVRequestResult(False, error="Fish returned invalid WAV audio", text=request_text)
                    if packed:
                        audio = normalize_wav(audio)
                else:
                    body = await resp.json()
                    status = body.get("base_resp") or {}
                    if status.get("status_code", 0) != 0:
                        return GSVRequestResult(False, error=f"SenseAudio status {status['status_code']}", text=request_text)
                    audio = bytes.fromhex((body.get("data") or {}).get("audio", ""))
                    if audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
                        return GSVRequestResult(False, error="SenseAudio returned invalid WAV audio", text=request_text)
                return GSVRequestResult(
                    ok=True,
                    data=audio,
                    text=request_text,
                )

        except ClientError as e:
            logger.error(f"[HTTP] 请求失败: {url} | {e}")
            return GSVRequestResult(False, error=str(e), text=request_text)

        except Exception as e:
            logger.exception(f"[HTTP] 未知异常: {url}")
            return GSVRequestResult(False, error=str(e), text=request_text)

    async def set_gpt_weights(self, path: str) -> GSVRequestResult:
        return await self._request(
            self.gpt_url,
            params={"weights_path": path},
        )

    async def set_sovits_weights(self, path: str) -> GSVRequestResult:
        return await self._request(
            self.sovits_url,
            params={"weights_path": path},
        )

    async def tts(self, params: dict) -> GSVRequestResult:
        voice_id = params.get("voice_id") or self.cfg.voice_id  # 人设自己的音色优先
        if self.cfg.backend == "fish":
            if not self.cfg.api_key:
                return GSVRequestResult(False, error="Fish API key is required")
            if self.cfg.model not in {"s2-pro", "s2.1-pro", "s2.1-pro-free"}:
                return GSVRequestResult(False, error="Select an explicit Fish S2 model, such as s2.1-pro-free")
            if voice_id:  # Fish 上存好的声音模型，填了就不用参考音频
                voice = {"reference_id": voice_id}
            elif not params.get("ref_audio_path") or not params.get("prompt_text"):
                return GSVRequestResult(False, error="Fish voice ID, or reference audio and transcript, is required")
            else:
                try:
                    voice = {"references": [{"audio": Path(params["ref_audio_path"]).read_bytes(), "text": params["prompt_text"]}]}
                except OSError as exc:
                    return GSVRequestResult(False, error=f"Fish reference audio unavailable: {exc}")
            text = normalize_fish_text(params.get("text", ""), keep_tags=not voice_id)
            logger.info(f"Fish 朗读：{text}")
            if not text:
                return GSVRequestResult(False, error="Fish text is empty")
            return await self._request(
                f"{self.base_url}/v1/tts", packed=True,
                payload={
                    "text": text,
                    **voice,
                    "format": "wav",
                    "temperature": params.get("temperature", 0.7),
                    "top_p": params.get("top_p", 0.7),
                    "prosody": {"speed": params.get("speed_factor", 1.0), "volume": 0, "normalize_loudness": True},
                    "latency": "normal",
                },
            )
        params = {k: v for k, v in params.items() if k != "voice_id"} | {"text": normalize_pause_tags(params.get("text", ""), self.cfg.backend == "senseaudio")}
        if self.cfg.backend == "senseaudio":
            if not self.cfg.api_key or not voice_id:
                return GSVRequestResult(False, error="SenseAudio API key and voice ID are required")
            return await self._request(
                f"{self.base_url}/v1/t2a_v2",
                payload={
                    "model": self.cfg.model,
                    "text": params.get("text", ""),
                    "stream": False,
                    "voice_setting": {"voice_id": voice_id, "speed": params.get("speed_factor", 1.0), "vol": 1, "pitch": 0},
                    "audio_setting": {"format": "wav", "sample_rate": 32000, "channel": 1},
                },
            )
        return await self._request(
            self.tts_url,
            params=params,
        )

    async def restart(self) -> GSVRequestResult:
        return await self._request(
            self.control_url,
            params={"command": "restart"},
        )
