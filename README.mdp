"""
Wrapper del cliente DeepSeek con gestión de sesiones y archivos.
Importa el motor desde la carpeta deepseekcli.
"""

import logging
import os
import sys
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

deepseekcli_path = project_root / "deepseekcli"
if not deepseekcli_path.exists():
    raise ImportError(f"No se encontró la carpeta 'deepseekcli' en: {deepseekcli_path}")

from deepseekcli import DeepSeekClient  # noqa: E402

from config import Config  # noqa: E402


class DeepSeekService:
    """Servicio singleton (thread-safe) para mantener el cliente DeepSeek."""

    _instance = None
    _init_lock = threading.Lock()
    _client = None
    _chat_semaphore: Optional[threading.Semaphore] = None

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is not None:
            return
        with self._init_lock:
            if self._client is not None:
                return
            try:
                token, cookies = Config.DEEPSEEK_TOKEN, Config.DEEPSEEK_COOKIES
                if not token or not cookies:
                    raise ValueError("Faltan DEEPSEEK_TOKEN y/o DEEPSEEK_COOKIES")
                login_dir = Path(Config.DEEPSEEK_LOGIN_DIR)
                self._client = DeepSeekClient(token=token, cookies=cookies, login_dir=login_dir)
                self._chat_semaphore = threading.Semaphore(Config.MAX_CONCURRENT_CHATS)
                logger.info("✅ Cliente DeepSeek inicializado correctamente")
            except Exception:
                logger.exception("❌ Error al inicializar el cliente DeepSeek")
                raise

    @property
    def client(self) -> DeepSeekClient:
        return self._client

    def create_session(self) -> str:
        try:
            session_id = self.client.create_chat_session()
            logger.info("✅ Sesión creada: %s", session_id)
            return session_id
        except Exception:
            logger.error("❌ Error al crear sesión", exc_info=True)
            raise

    def upload_file(self, file_path: str, thinking: bool = True) -> str:
        try:
            file_id = self.client.upload_file(file_path, thinking_enabled=thinking)
            logger.info("✅ Archivo subido: %s", file_id)
            return file_id
        except Exception:
            logger.error("❌ Error al subir archivo", exc_info=True)
            raise

    # ============================================================
    # HELPERS INTERNOS (no duplicar código)
    # ============================================================

    def _run_chat_in_thread(
        self,
        queue: Queue,
        target_func,
        *args,
        **kwargs,
    ) -> threading.Thread:
        """Ejecuta cualquier método del cliente en un hilo daemon y
        alimenta la cola con eventos (think, response, done, error)."""

        def worker():
            try:
                result = target_func(*args, **kwargs)
                # El cliente devuelve tuplas; desempaquetamos según el método.
                # NOTA: think/response YA se enviaron a la cola en tiempo real
                # a través de los callbacks on_think_chunk/on_response_chunk
                # (ver send_message/regenerate_message/continue_message más
                # abajo). Reenviarlos aquí de nuevo duplicaría todo el texto.
                # Solo queda comunicar que terminó, con el id del mensaje.
                if isinstance(result, tuple):
                    if len(result) == 4:  # (think, response, msg_id, is_incomplete)
                        _think, _response, msg_id, is_incomplete = result
                        queue.put(("done", {"msg_id": msg_id, "is_incomplete": is_incomplete}))
                    elif len(result) == 3:  # (think, response, msg_id) — legacy
                        _think, _response, msg_id = result
                        queue.put(("done", {"msg_id": msg_id, "is_incomplete": False}))
                    else:
                        queue.put(("error", f"Formato de respuesta inesperado: {result}"))
                else:
                    queue.put(("error", f"Formato de respuesta inesperado: {result}"))
            except Exception as e:
                logger.exception("❌ Error en el hilo de chat")
                queue.put(("error", str(e)))
            finally:
                self._chat_semaphore.release()

        acquired = self._chat_semaphore.acquire(timeout=Config.CHAT_TIMEOUT_SECONDS)
        if not acquired:
            queue.put(("error", "Servidor saturado, intenta de nuevo en unos segundos."))
            return None

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread

    def _yield_from_queue(
        self,
        queue: Queue,
        timeout: int,
    ) -> Generator[Dict[str, Any], None, None]:
        """Consume eventos de la cola hasta recibir done/error/timeout."""
        deadline_hit = False
        while True:
            try:
                event_type, data = queue.get(timeout=timeout)
            except Empty:
                deadline_hit = True
                yield {"type": "error", "data": "Tiempo de espera agotado esperando respuesta del backend."}
                break

            if event_type == "done":
                yield {"type": "done", "data": data}
                break
            elif event_type == "error":
                yield {"type": "error", "data": data}
                break
            else:
                yield {"type": event_type, "data": data}

        if deadline_hit:
            # Dar tiempo al hilo para terminar limpiamente
            pass

    # ============================================================
    # SEND MESSAGE (chat normal)
    # ============================================================

    def send_message(
        self,
        session_id: str,
        prompt: str,
        parent_message_id: Optional[int] = None,
        ref_file_ids: Optional[List[str]] = None,
        thinking_enabled: bool = True,
        search_enabled: bool = True,
        model_type: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Envía un mensaje y devuelve un generador de eventos (streaming interno)."""
        if Config.LOG_PROMPT_CONTENT:
            logger.info("📤 session=%s thinking=%s search=%s prompt=%r", session_id, thinking_enabled, search_enabled, prompt[:100])
        else:
            logger.info("📤 session=%s thinking=%s search=%s len(prompt)=%d", session_id, thinking_enabled, search_enabled, len(prompt))

        queue: Queue = Queue()

        def on_think(chunk: str):
            queue.put(("think", chunk))

        def on_response(chunk: str):
            queue.put(("response", chunk))

        def on_msg_id(msg_id: int):
            queue.put(("msg_id", msg_id))

        def target():
            return self.client.chat(
                prompt=prompt,
                session_id=session_id,
                parent_message_id=parent_message_id,
                ref_file_ids=ref_file_ids,
                stream=True,
                thinking_enabled=thinking_enabled,
                search_enabled=search_enabled,
                print_output=False,
                on_think_chunk=on_think,
                on_response_chunk=on_response,
                on_message_id=on_msg_id,
                save_history=True,
            )

        thread = self._run_chat_in_thread(queue, target)
        if thread is None:
            yield {"type": "error", "data": "Servidor saturado, intenta de nuevo en unos segundos."}
            return

        yield from self._yield_from_queue(queue, Config.CHAT_TIMEOUT_SECONDS)

    # ============================================================
    # REGENERATE
    # ============================================================

    def regenerate_message(
        self,
        session_id: str,
        child_message_id: int,
        thinking_enabled: bool = True,
        search_enabled: bool = True,
        user_options: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Regenera una respuesta existente. Devuelve generador de eventos."""
        logger.info("🔄 Regenerando message_id=%s en session=%s", child_message_id, session_id)

        queue: Queue = Queue()

        def on_think(chunk: str):
            queue.put(("think", chunk))

        def on_response(chunk: str):
            queue.put(("response", chunk))

        def on_msg_id(msg_id: int):
            queue.put(("msg_id", msg_id))

        def target():
            return self.client.regenerate(
                session_id=session_id,
                child_message_id=child_message_id,
                stream=True,
                thinking_enabled=thinking_enabled,
                search_enabled=search_enabled,
                user_options=user_options,
                print_output=False,
                on_think_chunk=on_think,
                on_response_chunk=on_response,
                on_message_id=on_msg_id,
                save_history=True,
            )

        thread = self._run_chat_in_thread(queue, target)
        if thread is None:
            yield {"type": "error", "data": "Servidor saturado, intenta de nuevo en unos segundos."}
            return

        yield from self._yield_from_queue(queue, Config.CHAT_TIMEOUT_SECONDS)

    # ============================================================
    # STOP STREAM
    # ============================================================

    def stop_message_stream(self, session_id: str, message_id: int) -> Dict[str, Any]:
        """Detiene la generación en curso de un mensaje específico."""
        logger.info("🛑 Deteniendo stream message_id=%s en session=%s", message_id, session_id)
        try:
            result = self.client.stop_stream(session_id=session_id, message_id=message_id)
            return {"success": True, "data": result}
        except Exception as e:
            logger.exception("❌ Error al detener stream")
            return {"success": False, "error": str(e)}

    # ============================================================
    # CONTINUE GENERATION
    # ============================================================

    def continue_message(
        self,
        session_id: str,
        message_id: int,
        fallback_to_resume: bool = True,
    ) -> Generator[Dict[str, Any], None, None]:
        """Continúa una respuesta que quedó INCOMPLETE."""
        logger.info("▶️ Continuando message_id=%s en session=%s", message_id, session_id)

        queue: Queue = Queue()

        def on_think(chunk: str):
            queue.put(("think", chunk))

        def on_response(chunk: str):
            queue.put(("response", chunk))

        def on_msg_id(msg_id: int):
            queue.put(("msg_id", msg_id))

        def target():
            return self.client.continue_generation(
                session_id=session_id,
                message_id=message_id,
                fallback_to_resume=fallback_to_resume,
                stream=True,
                print_output=False,
                on_think_chunk=on_think,
                on_response_chunk=on_response,
                on_message_id=on_msg_id,
                save_history=False,
            )

        thread = self._run_chat_in_thread(queue, target)
        if thread is None:
            yield {"type": "error", "data": "Servidor saturado, intenta de nuevo en unos segundos."}
            return

        yield from self._yield_from_queue(queue, Config.CHAT_TIMEOUT_SECONDS)
