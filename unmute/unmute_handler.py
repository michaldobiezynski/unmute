import asyncio
import math
from difflib import SequenceMatcher
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import websockets
from fastrtc import (
    AdditionalOutputs,
    AsyncStreamHandler,
    CloseStream,
    audio_to_float32,
    wait_for_item,
)
from pydantic import BaseModel

import unmute.openai_realtime_api_events as ora
from unmute import metrics as mt
from unmute.audio_input_override import AudioInputOverride
from unmute.exceptions import make_ora_error
from unmute.kyutai_constants import (
    FRAME_TIME_SEC,
    RECORDINGS_DIR,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
)
from unmute.llm.chatbot import Chatbot
from unmute.llm.llm_utils import (
    INTERRUPTION_CHAR,
    USER_SILENCE_MARKER,
    VLLMStream,
    get_openai_client,
    rechunk_to_words,
)
from unmute.quest_manager import Quest, QuestManager
from unmute.recorder import Recorder
from unmute.service_discovery import find_instance
from unmute.stt.speech_to_text import SpeechToText, STTMarkerMessage
from unmute.timer import Stopwatch
from unmute.tts.text_to_speech import (
    TextToSpeech,
    TTSAudioMessage,
    TTSClientEosMessage,
    TTSTextMessage,
)
from unmute.tts.voices import TimingConfig, VoiceList

# TTS_DEBUGGING_TEXT: str | None = "What's 'Hello world'?"
# TTS_DEBUGGING_TEXT: str | None = "What's the difference between a bagel and a donut?"
TTS_DEBUGGING_TEXT = None

# AUDIO_INPUT_OVERRIDE: Path | None = Path.home() / "audio/dog-or-cat-3.mp3"
AUDIO_INPUT_OVERRIDE: Path | None = None
DEBUG_PLOT_HISTORY_SEC = 10.0

# Default timing configurations (can be overridden per-character in voices.yaml)
DEFAULT_USER_SILENCE_TIMEOUT = 7.0
DEFAULT_PAUSE_THRESHOLD = 0.6
DEFAULT_INTERRUPTION_THRESHOLD = 0.4

FIRST_MESSAGE_TEMPERATURE = 0.7
FURTHER_MESSAGES_TEMPERATURE = 0.3
# For this much time, the VAD does not interrupt the bot. This is needed because at
# least on Mac, the echo cancellation takes a while to kick in, at the start, so the ASR
# sometimes hears a bit of the TTS audio and interrupts the bot. Only happens on the
# first message.
# A word from the ASR can still interrupt the bot.
UNINTERRUPTIBLE_BY_VAD_TIME_SEC = 3

# Minimum similarity threshold for considering text as "reading back" the assistant's response
# Range: 0.0 (completely different) to 1.0 (identical)
TEXT_SIMILARITY_THRESHOLD = 0.75

logger = getLogger(__name__)


def normalize_text_for_comparison(text: str) -> str:
    """Normalize text for similarity comparison by removing punctuation and converting to lowercase."""
    import re
    # Remove punctuation and extra whitespace, convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute similarity ratio between two texts using SequenceMatcher.
    
    Returns a value between 0.0 and 1.0, where 1.0 means identical texts.
    """
    normalized1 = normalize_text_for_comparison(text1)
    normalized2 = normalize_text_for_comparison(text2)
    
    if not normalized1 or not normalized2:
        return 0.0
    
    return SequenceMatcher(None, normalized1, normalized2).ratio()


def is_text_reading_back_response(user_text: str, last_assistant_text: str) -> bool:
    """Check if the user is reading back the assistant's last response.
    
    Returns True if user_text is similar enough to last_assistant_text.
    """
    if not last_assistant_text or not user_text:
        return False
    
    # Check both full match and substring match (user might read part of the response)
    similarity = compute_text_similarity(user_text, last_assistant_text)
    
    # Also check if user text is reading a substring of the assistant's response
    # This handles cases where user reads the beginning of a long response
    words_user = normalize_text_for_comparison(user_text).split()
    words_assistant = normalize_text_for_comparison(last_assistant_text).split()
    
    if len(words_user) >= 3 and len(words_assistant) >= len(words_user):
        # Check if user text matches the beginning of assistant text
        start_similarity = compute_text_similarity(
            " ".join(words_user),
            " ".join(words_assistant[:len(words_user)])
        )
        if start_similarity > TEXT_SIMILARITY_THRESHOLD:
            return True
    
    return similarity > TEXT_SIMILARITY_THRESHOLD

HandlerOutput = (
    tuple[int, np.ndarray] | AdditionalOutputs | ora.ServerEvent | CloseStream
)


class GradioUpdate(BaseModel):
    chat_history: list[dict[str, str]]
    debug_dict: dict[str, Any]
    debug_plot_data: list[dict]


class UnmuteHandler(AsyncStreamHandler):
    def __init__(self) -> None:
        super().__init__(
            input_sample_rate=SAMPLE_RATE,
            # IMPORTANT! If set to a higher value, will lead to choppy audio. 🤷‍♂️
            output_frame_size=480,
            output_sample_rate=SAMPLE_RATE,
        )
        self.n_samples_received = 0  # Used for measuring time
        self.output_queue: asyncio.Queue[HandlerOutput] = asyncio.Queue()
        self.recorder = Recorder(RECORDINGS_DIR) if RECORDINGS_DIR else None

        self.quest_manager = QuestManager()

        self.stt_last_message_time: float = 0
        self.stt_end_of_flush_time: float | None = None
        self.stt_flush_timer = Stopwatch()
        self.current_user_message_from_stt: str = ""  # Track the current user message being built

        self.tts_voice: str | None = None  # Stored separately because TTS is restarted
        self.tts_output_stopwatch = Stopwatch()
        self.text_only_mode: bool = False  # If True, skip TTS and only send text
        self.timing_config: TimingConfig = TimingConfig()  # Voice-specific timing config

        self.chatbot = Chatbot()
        self.openai_client = get_openai_client()
        self.last_complete_assistant_message: str = ""  # Track last assistant response for echo detection

        self.turn_transition_lock = asyncio.Lock()

        self.debug_dict: dict[str, Any] = {
            "timing": {},
            "connection": {},
            "chatbot": {},
        }
        self.debug_plot_data: list[dict] = []
        self.last_additional_output_update = self.audio_received_sec()

        if AUDIO_INPUT_OVERRIDE is not None:
            self.audio_input_override = AudioInputOverride(AUDIO_INPUT_OVERRIDE)
        else:
            self.audio_input_override = None

    async def cleanup(self):
        if self.recorder is not None:
            await self.recorder.shutdown()

    @property
    def stt(self) -> SpeechToText | None:
        try:
            quest = self.quest_manager.quests["stt"]
        except KeyError:
            return None
        return cast(Quest[SpeechToText], quest).get_nowait()

    @property
    def tts(self) -> TextToSpeech | None:
        try:
            quest = self.quest_manager.quests["tts"]
        except KeyError:
            return None
        return cast(Quest[TextToSpeech], quest).get_nowait()

    def get_gradio_update(self):
        self.debug_dict["conversation_state"] = self.chatbot.conversation_state()
        self.debug_dict["connection"]["stt"] = self.stt.state() if self.stt else "none"
        self.debug_dict["connection"]["tts"] = self.tts.state() if self.tts else "none"
        self.debug_dict["tts_voice"] = self.tts.voice if self.tts else "none"
        self.debug_dict["stt_pause_prediction"] = (
            self.stt.pause_prediction.value if self.stt else -1
        )
        self.debug_dict["timing_config"] = {
            "user_silence_timeout": (
                self.timing_config.user_silence_timeout
                if self.timing_config.user_silence_timeout is not None
                else DEFAULT_USER_SILENCE_TIMEOUT
            ),
            "pause_threshold": (
                self.timing_config.pause_threshold
                if self.timing_config.pause_threshold is not None
                else DEFAULT_PAUSE_THRESHOLD
            ),
            "interruption_threshold": (
                self.timing_config.interruption_threshold
                if self.timing_config.interruption_threshold is not None
                else DEFAULT_INTERRUPTION_THRESHOLD
            ),
        }

        # This gets verbose
        # cutoff_time = self.audio_received_sec() - DEBUG_PLOT_HISTORY_SEC
        # self.debug_plot_data = [x for x in self.debug_plot_data if x["t"] > cutoff_time]

        return AdditionalOutputs(
            GradioUpdate(
                chat_history=[
                    # Not trying to hide the system prompt, just making it less verbose
                    m
                    for m in self.chatbot.chat_history
                    if m["role"] != "system"
                ],
                debug_dict=self.debug_dict,
                debug_plot_data=[],
            )
        )

    async def add_chat_message_delta(
        self,
        delta: str,
        role: Literal["user", "assistant"],
        generating_message_i: int | None = None,  # Avoid race conditions
    ):
        is_new_message = await self.chatbot.add_chat_message_delta(
            delta, role, generating_message_i=generating_message_i
        )

        return is_new_message

    async def _generate_response(self):
        # Empty message to signal we've started responding.
        # Do it here in the lock to avoid race conditions
        await self.add_chat_message_delta("", "assistant")
        quest = Quest.from_run_step("llm", self._generate_response_task)
        await self.quest_manager.add(quest)

    async def _generate_response_task(self):
        generating_message_i = len(self.chatbot.chat_history)

        await self.output_queue.put(
            ora.ResponseCreated(
                response=ora.Response(
                    status="in_progress",
                    voice=self.tts_voice or "missing",
                    chat_history=self.chatbot.chat_history,
                )
            )
        )

        llm_stopwatch = Stopwatch()

        # Only start TTS if not in text-only mode
        quest = None if self.text_only_mode else await self.start_up_tts(generating_message_i)
        
        # Get LLM parameters from instructions if available
        instructions = self.chatbot.get_instructions()
        temperature = FIRST_MESSAGE_TEMPERATURE if generating_message_i == 2 else FURTHER_MESSAGES_TEMPERATURE
        max_tokens = None
        top_p = None
        stop = None
        
        if instructions and hasattr(instructions, 'temperature'):
            temperature = instructions.temperature
        if instructions and hasattr(instructions, 'max_tokens'):
            max_tokens = instructions.max_tokens
        if instructions and hasattr(instructions, 'top_p'):
            top_p = instructions.top_p
        if instructions and hasattr(instructions, 'stop'):
            stop = instructions.stop
        
        llm = VLLMStream(
            # if generating_message_i is 2, then we have a system prompt + an empty
            # assistant message signalling that we are generating a response.
            self.openai_client,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
        )

        messages = self.chatbot.preprocessed_messages()

        self.tts_output_stopwatch = Stopwatch(autostart=False)
        tts = None

        response_words = []
        error_from_tts = False
        time_to_first_token = None
        num_words_sent = sum(
            len(message.get("content", "").split()) for message in messages
        )
        mt.VLLM_SENT_WORDS.inc(num_words_sent)
        mt.VLLM_REQUEST_LENGTH.observe(num_words_sent)
        mt.VLLM_ACTIVE_SESSIONS.inc()

        try:
            async for delta in rechunk_to_words(llm.chat_completion(messages)):
                await self.output_queue.put(
                    ora.UnmuteResponseTextDeltaReady(delta=delta)
                )

                mt.VLLM_RECV_WORDS.inc()
                response_words.append(delta)

                if time_to_first_token is None:
                    time_to_first_token = llm_stopwatch.time()
                    self.debug_dict["timing"]["to_first_token"] = time_to_first_token
                    mt.VLLM_TTFT.observe(time_to_first_token)
                    logger.info("Sending first word to TTS: %s", delta)

                # In text-only mode, send text deltas directly
                if self.text_only_mode:
                    await self.output_queue.put(ora.ResponseTextDelta(delta=delta))
                    await self.add_chat_message_delta(
                        delta,
                        "assistant",
                        generating_message_i=generating_message_i,
                    )
                else:
                    assert quest is not None, "quest should not be None when not in text-only mode"
                    self.tts_output_stopwatch.start_if_not_started()
                    try:
                        tts = await quest.get()
                    except Exception:
                        error_from_tts = True
                        raise

                    if len(self.chatbot.chat_history) > generating_message_i:
                        break  # We've been interrupted

                    assert isinstance(delta, str)  # make Pyright happy
                    await tts.send(delta)

            # Store the complete assistant response for echo detection
            complete_response = "".join(response_words)
            self.last_complete_assistant_message = complete_response
            
            await self.output_queue.put(
                # The words include the whitespace, so no need to add it here
                ora.ResponseTextDone(text=complete_response)
            )

            if tts is not None:
                logger.info("Sending TTS EOS.")
                await tts.send(TTSClientEosMessage())
            
            # In text-only mode, signal the end of response differently
            if self.text_only_mode:
                await self.output_queue.put(ora.ResponseAudioDone())
                await self.add_chat_message_delta("", "user")
                self.waiting_for_user_start_time = self.audio_received_sec()
        except asyncio.CancelledError:
            mt.VLLM_INTERRUPTS.inc()
            raise
        except Exception:
            if not error_from_tts:
                mt.VLLM_HARD_ERRORS.inc()
            raise
        finally:
            logger.info("End of VLLM, after %d words.", len(response_words))
            mt.VLLM_ACTIVE_SESSIONS.dec()
            mt.VLLM_REPLY_LENGTH.observe(len(response_words))
            mt.VLLM_GEN_DURATION.observe(llm_stopwatch.time())

    def audio_received_sec(self) -> float:
        """How much audio has been received in seconds. Used instead of time.time().

        This is so that we aren't tied to real-time streaming.
        """
        return self.n_samples_received / self.input_sample_rate

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        stt = self.stt
        assert stt is not None
        sr = frame[0]
        assert sr == self.input_sample_rate

        assert frame[1].shape[0] == 1  # Mono
        array = frame[1][0]

        self.n_samples_received += array.shape[0]

        # If this doesn't update, it means the receive loop isn't running because
        # the process is busy with something else, which is bad.
        self.debug_dict["last_receive_time"] = self.audio_received_sec()
        float_audio = audio_to_float32(array)

        self.debug_plot_data.append(
            {
                "t": self.audio_received_sec(),
                "amplitude": float(np.sqrt((float_audio**2).mean())),
                "pause_prediction": stt.pause_prediction.value,
            }
        )

        if self.chatbot.conversation_state() == "bot_speaking":
            # Periodically update this not to trigger the "long silence" accidentally.
            self.waiting_for_user_start_time = self.audio_received_sec()

        if TTS_DEBUGGING_TEXT is not None:
            assert self.audio_input_override is None, (
                "Can't use both TTS_DEBUGGING_TEXT and audio input override."
            )

            # Debugging mode: always send a fixed string when it's the user's turn.
            if self.chatbot.conversation_state() == "waiting_for_user":
                logger.info("Using TTS debugging text. Ignoring microphone.")
                self.chatbot.chat_history.append(
                    {"role": "user", "content": TTS_DEBUGGING_TEXT}
                )
                await self._generate_response()
            return

        if (
            len(self.chatbot.chat_history) == 1
            # Wait until the instructions are updated. A bit hacky
            and self.chatbot.get_instructions() is not None
        ):
            logger.info("Generating initial response.")
            await self._generate_response()

        if self.audio_input_override is not None:
            frame = (frame[0], self.audio_input_override.override(frame[1]))

        if self.chatbot.conversation_state() == "user_speaking":
            self.debug_dict["timing"] = {}

        await stt.send_audio(array)
        if self.stt_end_of_flush_time is None:
            await self.detect_long_silence()

            if self.determine_pause():
                logger.info("Pause detected")
                await self.output_queue.put(ora.InputAudioBufferSpeechStopped())

                self.stt_end_of_flush_time = stt.current_time + stt.delay_sec
                self.stt_flush_timer = Stopwatch()
                num_frames = (
                    int(math.ceil(stt.delay_sec / FRAME_TIME_SEC)) + 1
                )  # some safety margin.
                zero = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
                for _ in range(num_frames):
                    await stt.send_audio(zero)
            elif (
                self.chatbot.conversation_state() == "bot_speaking"
                and stt.pause_prediction.value
                < (
                    self.timing_config.interruption_threshold
                    if self.timing_config.interruption_threshold is not None
                    else DEFAULT_INTERRUPTION_THRESHOLD
                )
                and self.audio_received_sec() > UNINTERRUPTIBLE_BY_VAD_TIME_SEC
            ):
                logger.info("Interruption by STT-VAD")
                await self.interrupt_bot()
                await self.add_chat_message_delta("", "user")
        else:
            # We do not try to detect interruption here, the STT would be processing
            # a chunk full of 0, so there is little chance the pause score would indicate an interruption.
            if stt.current_time > self.stt_end_of_flush_time:
                self.stt_end_of_flush_time = None
                elapsed = self.stt_flush_timer.time()
                rtf = stt.delay_sec / elapsed
                logger.info(
                    "Flushing finished, took %.1f ms, RTF: %.1f", elapsed * 1000, rtf
                )
                await self._generate_response()

    def determine_pause(self) -> bool:
        stt = self.stt
        if stt is None:
            return False
        if self.chatbot.conversation_state() != "user_speaking":
            return False

        # This is how much wall clock time has passed since we received the last ASR
        # message. Assumes the ASR connection is healthy, so that stt.sent_samples is up
        # to date.
        time_since_last_message = (
            stt.sent_samples / self.input_sample_rate
        ) - self.stt_last_message_time
        self.debug_dict["time_since_last_message"] = time_since_last_message

        # Use voice-specific threshold or default
        pause_threshold = (
            self.timing_config.pause_threshold
            if self.timing_config.pause_threshold is not None
            else DEFAULT_PAUSE_THRESHOLD
        )

        if stt.pause_prediction.value > pause_threshold:
            self.debug_dict["timing"]["pause_detection"] = time_since_last_message
            return True
        else:
            return False

    async def emit(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> HandlerOutput | None:
        output_queue_item = await wait_for_item(self.output_queue)

        if output_queue_item is not None:
            return output_queue_item
        else:
            if self.last_additional_output_update < self.audio_received_sec() - 1:
                # If we have nothing to emit, at least update the debug dict.
                # Don't update too often for performance reasons
                self.last_additional_output_update = self.audio_received_sec()
                return self.get_gradio_update()
            else:
                return None

    def copy(self):
        return UnmuteHandler()

    def load_timing_config_for_voice(self, voice_path: str) -> TimingConfig:
        """Load timing configuration for the specified voice from voices.yaml."""
        try:
            voice_list = VoiceList()
            for voice in voice_list.voices:
                if voice.source.path_on_server == voice_path:
                    if voice.timing:
                        logger.info(
                            f"Loaded custom timing config for voice: {voice.name or voice_path}"
                        )
                        return voice.timing
                    break
            logger.info(f"Using default timing config for voice: {voice_path}")
            return TimingConfig()
        except Exception as e:
            logger.warning(f"Error loading timing config: {e}, using defaults")
            return TimingConfig()

    async def __aenter__(self) -> None:
        await self.quest_manager.__aenter__()

    async def start_up(self):
        await self.start_up_stt()
        self.waiting_for_user_start_time = self.audio_received_sec()

    async def __aexit__(self, *exc: Any) -> None:
        return await self.quest_manager.__aexit__(*exc)

    async def start_up_stt(self):
        async def _init() -> SpeechToText:
            return await find_instance("stt", SpeechToText)

        async def _run(stt: SpeechToText):
            await self._stt_loop(stt)

        async def _close(stt: SpeechToText):
            await stt.shutdown()

        quest = await self.quest_manager.add(Quest("stt", _init, _run, _close))
        # We want to be sure to have the STT before starting anything.
        await quest.get()

    async def _stt_loop(self, stt: SpeechToText):
        try:
            async for data in stt:
                if isinstance(data, STTMarkerMessage):
                    # Ignore the marker messages
                    continue

                await self.output_queue.put(
                    ora.ConversationItemInputAudioTranscriptionDelta(
                        delta=data.text,
                        start_time=data.start_time,
                    )
                )

                # The STT sends an empty string as the first message, but we
                # don't want to add that because it can trigger a pause even
                # if the user hasn't started speaking yet.
                if data.text == "":
                    # Reset the current user message when STT sends empty string
                    self.current_user_message_from_stt = ""
                    continue

                # Build up the current user message
                if data.text:
                    # Add spacing if needed
                    if self.current_user_message_from_stt and not self.current_user_message_from_stt[-1].isspace():
                        self.current_user_message_from_stt += " "
                    self.current_user_message_from_stt += data.text

                # Check if user is reading back the assistant's response
                if is_text_reading_back_response(
                    self.current_user_message_from_stt,
                    self.last_complete_assistant_message
                ):
                    logger.info(
                        "Detected user reading back assistant response. "
                        f"User text: '{self.current_user_message_from_stt[:100]}...' "
                        f"Assistant text: '{self.last_complete_assistant_message[:100]}...'"
                    )
                    # Skip processing this speech - don't interrupt, don't add to chat
                    continue

                if self.chatbot.conversation_state() == "bot_speaking":
                    logger.info("STT-based interruption")
                    await self.interrupt_bot()
                    # Reset current user message on interruption
                    self.current_user_message_from_stt = ""

                self.stt_last_message_time = data.start_time
                is_new_message = await self.add_chat_message_delta(data.text, "user")
                if is_new_message:
                    # Ensure we don't stop after the first word if the VAD didn't have
                    # time to react.
                    stt.pause_prediction.value = 0.0
                    await self.output_queue.put(ora.InputAudioBufferSpeechStarted())
                    # Reset the accumulated message on new message
                    self.current_user_message_from_stt = ""
        except websockets.ConnectionClosed:
            logger.info("STT connection closed while receiving messages.")

    async def start_up_tts(self, generating_message_i: int) -> Quest[TextToSpeech]:
        async def _init() -> TextToSpeech:
            factory = partial(
                TextToSpeech,
                recorder=self.recorder,
                get_time=self.audio_received_sec,
                voice=self.tts_voice,
            )
            sleep_time = 0.05
            sleep_growth = 1.5
            max_sleep = 1.0
            trials = 5
            for trial in range(trials):
                try:
                    tts = await find_instance("tts", factory)
                except Exception:
                    if trial == trials - 1:
                        raise
                    logger.warning("Will sleep for %.4f sec", sleep_time)
                    await asyncio.sleep(sleep_time)
                    sleep_time = min(max_sleep, sleep_time * sleep_growth)
                    error = make_ora_error(
                        type="warning",
                        message="Looking for the resources, expect some latency.",
                    )
                    await self.output_queue.put(error)
                else:
                    return tts
            raise AssertionError("Too many unexpected packets.")

        async def _run(tts: TextToSpeech):
            await self._tts_loop(tts, generating_message_i)

        async def _close(tts: TextToSpeech):
            await tts.shutdown()

        return await self.quest_manager.add(Quest("tts", _init, _run, _close))

    async def _tts_loop(self, tts: TextToSpeech, generating_message_i: int):
        # On interruption, we swap the output queue. This will ensure that this worker
        # can never accidentally push to the new queue if it's interrupted.
        output_queue = self.output_queue
        try:
            audio_started = None

            async for message in tts:
                if audio_started is not None:
                    time_since_start = self.audio_received_sec() - audio_started
                    time_received = tts.received_samples / self.input_sample_rate
                    time_received_yielded = (
                        tts.received_samples_yielded / self.input_sample_rate
                    )
                    assert self.input_sample_rate == SAMPLE_RATE
                    self.debug_dict["tts_throughput"] = {
                        "time_received": round(time_received, 2),
                        "time_received_yielded": round(time_received_yielded, 2),
                        "time_since_start": round(time_since_start, 2),
                        "ratio": round(
                            time_received_yielded / (time_since_start + 0.01), 2
                        ),
                    }

                if len(self.chatbot.chat_history) > generating_message_i:
                    break

                if isinstance(message, TTSAudioMessage):
                    t = self.tts_output_stopwatch.stop()
                    if t is not None:
                        self.debug_dict["timing"]["tts_audio"] = t

                    audio = np.array(message.pcm, dtype=np.float32)
                    assert self.output_sample_rate == SAMPLE_RATE

                    await output_queue.put((SAMPLE_RATE, audio))

                    if audio_started is None:
                        audio_started = self.audio_received_sec()
                elif isinstance(message, TTSTextMessage):
                    await output_queue.put(ora.ResponseTextDelta(delta=message.text))
                    await self.add_chat_message_delta(
                        message.text,
                        "assistant",
                        generating_message_i=generating_message_i,
                    )
                else:
                    logger.warning("Got unexpected message from TTS: %s", message.type)

        except websockets.ConnectionClosedError as e:
            logger.error(f"TTS connection closed with an error: {e}")

        # Push some silence to flush the Opus state.
        # Not sure that this is actually needed.
        await output_queue.put(
            (SAMPLE_RATE, np.zeros(SAMPLES_PER_FRAME, dtype=np.float32))
        )

        message = self.chatbot.last_message("assistant")
        if message is None:
            logger.warning("No message to send in TTS shutdown.")
            message = ""

        # It's convenient to have the whole chat history available in the client
        # after the response is done, so send the "gradio update"
        await self.output_queue.put(self.get_gradio_update())
        await self.output_queue.put(ora.ResponseAudioDone())

        # Signal that the turn is over by adding an empty message.
        await self.add_chat_message_delta("", "user")

        await asyncio.sleep(1)
        await self.check_for_bot_goodbye()
        self.waiting_for_user_start_time = self.audio_received_sec()

    async def interrupt_bot(self):
        if self.chatbot.conversation_state() != "bot_speaking":
            raise RuntimeError(
                "Can't interrupt bot when conversation state is "
                f"{self.chatbot.conversation_state()}"
            )

        await self.add_chat_message_delta(INTERRUPTION_CHAR, "assistant")

        if self._clear_queue is not None:
            # Clear any audio queued up by FastRTC's emit().
            # Not sure under what circumstatnces this is None.
            self._clear_queue()
        self.output_queue = asyncio.Queue()  # Clear our own queue too

        # Push some silence to flush the Opus state.
        # Not sure that this is actually needed.
        await self.output_queue.put(
            (SAMPLE_RATE, np.zeros(SAMPLES_PER_FRAME, dtype=np.float32))
        )

        await self.output_queue.put(ora.UnmuteInterruptedByVAD())

        await self.quest_manager.remove("tts")
        await self.quest_manager.remove("llm")

    async def check_for_bot_goodbye(self):
        last_assistant_message = next(
            (
                msg
                for msg in reversed(self.chatbot.chat_history)
                if msg["role"] == "assistant"
            ),
            {"content": ""},
        )["content"]

        # Using function calling would be a more robust solution, but it would make it
        # harder to swap LLMs.
        if last_assistant_message.lower().endswith("bye!"):
            await self.output_queue.put(
                CloseStream("The assistant ended the conversation. Bye!")
            )

    async def detect_long_silence(self):
        """Handle situations where the user doesn't answer for a while."""
        # Use voice-specific timeout or default
        silence_timeout = (
            self.timing_config.user_silence_timeout
            if self.timing_config.user_silence_timeout is not None
            else DEFAULT_USER_SILENCE_TIMEOUT
        )

        if (
            self.chatbot.conversation_state() == "waiting_for_user"
            and (self.audio_received_sec() - self.waiting_for_user_start_time)
            > silence_timeout
        ):
            # This will trigger pause detection because it changes the conversation
            # state to "user_speaking".
            # The system prompt has a rule that tells it how to handle the "..."
            # messages.
            logger.info("Long silence detected.")
            await self.add_chat_message_delta(USER_SILENCE_MARKER, "user")

    async def update_session(self, session: ora.SessionConfig):
        if session.instructions:
            self.chatbot.set_instructions(session.instructions)

        if session.voice:
            self.tts_voice = session.voice
            # Load timing configuration for the new voice
            self.timing_config = self.load_timing_config_for_voice(session.voice)

            # If STT has custom timing settings, update the pause prediction EMA
            if self.stt and (
                self.timing_config.attack_time is not None
                or self.timing_config.release_time is not None
            ):
                from unmute.stt.exponential_moving_average import ExponentialMovingAverage

                attack_time = (
                    self.timing_config.attack_time
                    if self.timing_config.attack_time is not None
                    else 0.01
                )
                release_time = (
                    self.timing_config.release_time
                    if self.timing_config.release_time is not None
                    else 0.01
                )
                current_value = self.stt.pause_prediction.value
                self.stt.pause_prediction = ExponentialMovingAverage(
                    attack_time=attack_time,
                    release_time=release_time,
                    initial_value=current_value,
                )
                logger.info(
                    f"Updated STT timing: attack={attack_time}, release={release_time}"
                )

        # Update text-only mode
        self.text_only_mode = session.text_only_mode
        if self.text_only_mode:
            logger.info("Text-only mode enabled, TTS will be skipped")

        # Reset echo detection tracking when session is updated
        self.last_complete_assistant_message = ""
        self.current_user_message_from_stt = ""

        if not session.allow_recording and self.recorder:
            await self.recorder.add_event("client", ora.SessionUpdate(session=session))
            await self.recorder.shutdown(keep_recording=False)
            self.recorder = None
            logger.info("Recording disabled for a session.")
