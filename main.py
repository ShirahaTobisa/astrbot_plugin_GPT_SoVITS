import asyncio
import base64
import json
import random
import re

from astrbot.api import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star
from astrbot.core import AstrBotConfig
from astrbot.core.agent.tool import ToolSet
from astrbot.core.message.components import Plain, Record
from astrbot.core.platform import AstrMessageEvent
from astrbot.core.persona_error_reply import resolve_event_conversation_persona_id

from .core.client import GSVApiClient, GSVRequestResult
from .core.config import PluginConfig
from .core.emotion import EmotionJudger
from .core.entry import EntryManager
from .core.local_data import LocalDataManager
from .core.service import GPTSoVITSService


# 用了这些办事工具（电脑操作、文件、执行代码都以 astrbot_ 开头，加上读网页）这一轮默认只发文字，
# 除非她最后回的是陪聊短句（见 chatty）；
# 记忆、生图、提醒、搜索、B 站这类陪聊里顺手用的工具照常朗读
WEB_READ_TOOLS = {"tavily_extract_web_page", "firecrawl_extract_web_page", "exa_get_contents"}


def chatty(zh: str) -> bool:
    """用过办事工具的那一轮里，这句回复像不像陪聊：短，且不带代码、链接、文件路径。像就照样朗读。"""
    return len(zh) <= 100 and not re.search(r"```|https?://|/[\w.-]+/|\\", zh)


class GPTSoVITSPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.cfg = PluginConfig(config, context)
        self.local_data = LocalDataManager(self.cfg)
        self.entry_mgr = EntryManager(self.cfg)
        self.client = GSVApiClient(self.cfg)
        self.judger = EmotionJudger(self.cfg)
        self.service = GPTSoVITSService(self.cfg, self.client, self.local_data)

    async def initialize(self):
        if self.cfg.enabled and self.cfg.client.backend == "gpt_sovits":
            await self.service.load_model()

    async def terminate(self):
        await self.client.close()

    @staticmethod
    def _to_record(res: GSVRequestResult) -> Record:
        if res.file_path:
            try:
                return Record.fromFileSystem(res.file_path)
            except Exception:
                logger.warning(f"无法读取文件：{res.file_path}, 已忽略")
                pass

        if not res.data:
            raise ValueError("无法获取结果数据")

        b64 = base64.b64encode(res.data).decode()
        return Record.fromBase64(b64)

    async def _resolve_persona_id(self, event, conversation=None):
        conversation_persona_id = getattr(conversation, "persona_id", None)
        if conversation is None:
            conversation_persona_id = await resolve_event_conversation_persona_id(event, self.context.conversation_manager)
        persona_id, _, _, _ = await self.context.persona_manager.resolve_selected_persona(
            umo=event.unified_msg_origin,
            conversation_persona_id=conversation_persona_id,
            platform_name=event.get_platform_name(),
            provider_settings=self.context.get_config(event.unified_msg_origin).get("provider_settings", {}),
        )
        return persona_id

    def _voice_of(self, persona_id):
        """这个人设念不念、用哪个音色和日语说话风格；不念返回 None。音色为 None 时用客户端配置里的。"""
        if not persona_id:
            return None
        if persona_id == self.cfg.auto.persona_id:
            return {"voice_id": None, "style": self.cfg.auto.japanese_style_prompt}
        for item in self.cfg.persona_voices or []:
            if item.get("persona_id") == persona_id and item.get("voice_id"):  # 还没填音色的先不念，免得借用默认音色
                return {"voice_id": item["voice_id"], "style": item.get("japanese_style_prompt")}
        return None

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def prepare_bilingual_message(self, event: AstrMessageEvent):
        if not self.cfg.enabled or not self.cfg.auto.bilingual:
            return
        persona_id = await self._resolve_persona_id(event)
        if self._voice_of(persona_id):
            event.set_extra("enable_streaming", False)

    @filter.on_llm_request(priority=110)
    async def prepare_bilingual_request(self, event, req):
        selected = await self._resolve_persona_id(event, req.conversation) if self.cfg.enabled else None
        voice = self._voice_of(selected)
        active = voice is not None
        event.set_extra("gsv_persona_active", active)
        event.set_extra("gsv_persona_id", selected if active else None)
        event.set_extra("gsv_pending_reply", active and self.cfg.auto.bilingual)
        event.set_extra("gsv_japanese_text", None)
        event.set_extra("quill_worldbook_persona_id", selected if active and self.cfg.auto.bilingual else None)
        if active and self.cfg.auto.bilingual:
            logger.info(f"TTS 已启用双语回复：生效人设 {selected}")
            # 工具直接发送会绕过语音处理；仅在本轮移除，不改全局工具列表。
            if req.func_tool:
                req.func_tool = ToolSet([
                    tool for tool in req.func_tool.tools
                    if tool.name != "send_message_to_user"
                ])
            req.system_prompt += (
                '\n\n最终回复由程序分开显示和朗读，只输出一个 JSON 对象。'
                '先根据本轮意图区分陪聊和办事，不需要额外调用分类工具。'
                '日常陪聊、关心、角色互动用 {"zh":"中文台词","ja":"对应的自然日语台词"}，'
                '两个字段表达相同内容、保持当前人设，不混合语言，不写动作或旁白。'
                '网页解读、参数比较、教程、代码或文件处理，以及这些任务的后续追问，'
                '用 {"zh":"完整的中文任务回复","ja":null}，不要生成日语或自动朗读。'
                '任务回复可在 zh 中保留必要的列表、代码、链接和引用，不受陪聊短句或字数要求限制；'
                '联网搜索、普通记忆回想、生图和生活提醒本身不等于办事，用搜到的信息聊天时照常按陪聊回复。'
                '工具调用过程中的进度不属于最终回复；不要给整个 JSON 加代码围栏。'
                '不附加状态栏或剧情走向选项。以下日语语气、情绪与朗读要求只适用于陪聊的 ja 字段：'
            )
            req.system_prompt += '\n\n' + (voice["style"] or '')

    @filter.on_using_llm_tool(priority=100)
    async def skip_task_speech(self, event, tool, tool_args):
        if event.get_extra("gsv_persona_active") and (tool.name.startswith("astrbot_") or tool.name in WEB_READ_TOOLS):
            if not event.get_extra("gsv_skip_tts"):
                logger.info(f"TTS 本轮仅发文字：任务工具 {tool.name}")
            event.set_extra("gsv_skip_tts", True)

    @filter.on_llm_response(priority=100)
    async def split_bilingual_response(self, event, response):
        if not event.get_extra("gsv_persona_active") or not self.cfg.auto.bilingual or response is None:
            return
        if getattr(response, "tools_call_name", None):
            return
        event.set_extra("gsv_pending_reply", False)
        event.set_extra("gsv_japanese_text", None)
        text = (response.completion_text or "").strip()
        if not text:
            return
        if text.startswith("```\n") or text.startswith("```json\n"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        # 模型偶尔先写一句中文、再给 JSON（新对话开头的预设对话是纯文本，会带偏格式），只取 JSON 那段
        prefix = ""
        if m := re.search(r'\{\s*"zh"\s*:', text):
            prefix = text[:m.start()].strip()
            text = text[m.start():text.rfind("}") + 1] or text[m.start():]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = self._loose_bilingual(text)
            if data is None:
                if prefix:  # JSON 坏了：只留前面那句中文去补译，不把半截 JSON 发出去
                    response.completion_text = prefix
                elif re.search(r"[぀-ヿ]", text):  # 没写 JSON、直接回了日语：日语留着朗读，翻成中文显示
                    event.set_extra("gsv_japanese_text", text)
                    response.completion_text = await self._translate_to_chinese(text, event.get_extra("gsv_persona_id")) or text
                logger.debug("TTS 回复未按双语格式返回，将在发送语音前补译")
                return
        if not isinstance(data, dict) or not isinstance(data.get("zh"), str) or not data["zh"].strip():
            return
        response.completion_text = data["zh"].strip()
        if "ja" in data and data["ja"] is None:
            event.set_extra("gsv_skip_tts", True)
            return
        if isinstance(data.get("ja"), str) and data["ja"].strip():
            event.set_extra("gsv_japanese_text", data["ja"].strip())
            # 走到这里还带着「只发文字」的标记，只可能是这一轮用过办事工具；她自己给了日语、话又像陪聊，就照样朗读
            if event.get_extra("gsv_skip_tts") and chatty(response.completion_text):
                logger.info("TTS 本轮用过任务工具，但回的是陪聊短句，照常朗读")
                event.set_extra("gsv_skip_tts", False)

    async def _send_text_first(self, event, chain):
        """插件提前发文字会绕过 AstrBot 自己的分段，这里照它的分段设置切开再发。"""
        seg = self.context.get_config(event.unified_msg_origin).get("platform_settings", {}).get("segmented_reply", {})
        text = "".join(c.text for c in chain if isinstance(c, Plain))
        if not seg.get("enable") or not all(isinstance(c, Plain) for c in chain) or len(text) > int(seg.get("words_count_threshold", 150)):
            await event.send(event.chain_result(chain))
            return
        try:
            parts = [p.strip() for p in re.findall(seg.get("regex") or r".*?[。？！~…]+|.+$", text, re.DOTALL | re.MULTILINE)]
            low, high = (float(x) for x in str(seg.get("interval", "1.5,3.5")).replace(" ", "").split(","))
        except (re.error, ValueError):
            parts, low, high = [], 0.0, 0.0
        for i, part in enumerate([p for p in parts if p] or [text]):
            if i:
                await asyncio.sleep(random.uniform(low, high))
            await event.send(event.chain_result([Plain(part)]))

    @staticmethod
    def _loose_bilingual(text: str) -> dict | None:
        """中文里夹了英文双引号或直接换行时 JSON 会坏，按字段位置把 zh / ja 拆出来。"""
        m = re.fullmatch(r'\{\s*"zh"\s*:\s*"(.*)"\s*,\s*"ja"\s*:\s*(?:null|"(.*)")\s*\}', text, re.S)
        if not m:
            return None
        unescape = lambda s: s.replace('\\"', '"').replace("\\n", "\n")
        return {"zh": unescape(m.group(1)), "ja": unescape(m.group(2)) if m.group(2) is not None else None}

    async def _translate_to_chinese(self, text: str, persona_id: str | None) -> str | None:
        logger.info(f"TTS 回复只有日语，翻成中文显示：日语 {len(text)} 字")
        try:
            async with asyncio.timeout(45):
                try:
                    persona_prompt = (await self.context.persona_manager.get_persona(persona_id)).system_prompt
                except ValueError:
                    persona_prompt = ""
                response = await self.context.llm_generate(
                    chat_provider_id=self.context.get_using_provider().meta().id,
                    system_prompt=(
                        "你就是下面人设里的角色。下面给你一段你自己刚说的日语，请换成你平时说中文的样子，"
                        "语气和意思都不变，不增减内容。只输出中文。\n\n" + persona_prompt
                    ),
                    prompt=text,
                )
            return (response.completion_text or "").strip() or None
        except Exception as exc:
            logger.warning(f"TTS 日语回译失败，保留日语原文：{type(exc).__name__}")
            return None

    async def _translate_to_japanese(self, event, text: str) -> str | None:
        logger.info(f"TTS 缺少日语，开始补译：中文 {len(text)} 字")
        try:
            async with asyncio.timeout(45):
                provider_id = await self.context.get_current_chat_provider_id(event.unified_msg_origin)
                persona_id = event.get_extra("gsv_persona_id") or self.cfg.auto.persona_id
                style = (self._voice_of(persona_id) or {}).get("style")
                try:
                    persona_prompt = (await self.context.persona_manager.get_persona(persona_id)).system_prompt
                except ValueError:
                    persona_prompt = ""
                # 带上人设和对方的话，说清楚中文是"你"说的：不带这些时补译会把台词当成对方的话去回应
                response = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    system_prompt=(
                        "你就是下面人设里的角色。下面先是你的人设，再给你对方刚才说的话和你刚用中文回的话。"
                        "请用你自己说日语的方式把你回的这句话说出来，用于语音朗读。"
                        "中文是你自己说的，不是对方说的，不要回应它，也不要增减意思，只把它换成日语。"
                        "只输出日语台词；如果这段话是列表、代码、链接、教程讲解之类的资料，不适合朗读，只输出 SKIP。"
                        "\n\n" + persona_prompt + "\n\n" + (style or "")
                    ),
                    prompt=f"对方刚才说：{event.message_str}\n\n你刚用中文回的话：{text}",
                )
            translated = (response.completion_text or "").strip()
            if not translated:
                raise ValueError("Empty translation")
            if translated.upper() == "SKIP":
                logger.info("TTS 补译判断是资料内容，本轮只发文字")
                return None
            return translated
        except Exception as exc:
            logger.warning(f"TTS 日语补译失败，已保留中文：{type(exc).__name__}")
            return None

    async def _get_emotion_params(
        self, event: AstrMessageEvent, text: str
    ) -> dict | None:
        entry = None

        if self.cfg.judge.enabled_llm:
            labels = self.entry_mgr.get_names()
            emotion = await self.judger.judge_emotion(event, text=text, labels=labels)
            if emotion:
                entry = self.entry_mgr.get_entry(emotion)

        if entry is None:
            entry = self.entry_mgr.match_entry(text)

        return entry.to_params() if entry else None

    @filter.on_decorating_result(priority=14)
    async def on_decorating_result(self, event: AstrMessageEvent):
        """消息入口"""
        if not self.cfg.enabled:
            return
        if event.get_extra("gsv_pending_reply"):
            # 最终回复还没出来就要发的模型文字，是调工具时顺手写的旁白（"某某让我画画"之类），不发
            result = event.get_result()
            if result and result.is_llm_result():
                result.chain.clear()
            return
        if event.get_extra("gsv_skip_tts"):
            return
        cfg = self.cfg.auto

        result = event.get_result()
        if not result:
            return
        chain = result.chain
        if not chain:
            return
        if cfg.only_llm_result and not result.is_llm_result():
            return
        if cfg.persona_id:
            active = event.get_extra("gsv_persona_active")
            if active is None:
                persona_id = await self._resolve_persona_id(event)
                active = self._voice_of(persona_id) is not None
                event.set_extra("gsv_persona_id", persona_id)
            if not active:
                return
        if random.random() > cfg.tts_prob:
            logger.debug("TTS 按语音概率设置跳过")
            return

        plain_texts = [seg.text for seg in chain if isinstance(seg, Plain)]
        if not plain_texts or any(isinstance(seg, Record) for seg in chain):
            return
        if not cfg.bilingual and len(plain_texts) != len(chain):
            return

        source_text = "\n".join(plain_texts).strip()
        if not source_text:
            return
        combined_text = source_text
        if cfg.bilingual:
            combined_text = event.get_extra("gsv_japanese_text")
            if not combined_text and len(source_text) > cfg.max_msg_len:
                logger.warning(f"TTS 待翻译文字超过 {cfg.max_msg_len} 字，保留中文")
                return
            # 补译和分段发文字同时进行，语音不用等文字发完才开始准备
            translating = None if combined_text else asyncio.create_task(self._translate_to_japanese(event, source_text))
            await self._send_text_first(event, list(chain))
            # 换新列表：原列表和模型回复共用，就地清空会让 AstrBot 以为回复为空，这一轮就不存聊天记录
            result.chain = []
            if translating:
                combined_text = await translating
            if not combined_text:
                return

        if len(combined_text) > cfg.max_msg_len:
            logger.warning(f"TTS 日语超过 {cfg.max_msg_len} 字，已保留文字")
            return

        try:
            params = await self._get_emotion_params(event, combined_text)
            voice = self._voice_of(event.get_extra("gsv_persona_id")) or {}
            res = await self.service.inference(combined_text, extra_params=params, voice_id=voice.get("voice_id"))
            if not res:
                logger.warning("TTS 合成未成功，已保留文字")
                return
            record = self._to_record(res)
        except Exception as exc:
            logger.warning(f"TTS 生成语音失败，已保留文字：{type(exc).__name__}")
            return
        result.chain = [record]
        logger.info(f"TTS 语音已生成，待发送：{len(combined_text)} 字")

    @filter.command("说", alias={"gsv", "GSV"})
    async def on_command(self, event: AstrMessageEvent):
        """说 <内容>, 直接调用GSV合成语音"""
        if not self.cfg.enabled:
            return

        text = event.message_str.partition(" ")[2]
        res = await self.service.inference(text)

        if not bool(res):
            yield event.plain_result(res.error)
            return

        yield event.chain_result([self._to_record(res)])

    @filter.command("重启GSV", alias={"重启gsv"})
    async def tts_control(self, event: AstrMessageEvent):
        """重启GPT_SoVITS"""
        if not self.cfg.enabled:
            return
        if self.cfg.client.backend != "gpt_sovits":
            yield event.plain_result("当前使用云端语音服务，无需重启本地模型。")
            return
        yield event.plain_result("重启TTS中...(报错信息请忽略，等待一会即可完成重启)")
        await self.service.restart()
