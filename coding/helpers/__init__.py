from .selector import Selector
from .parser import *
from .cosine import *
from .forwards import *
from .fim import *
import time
import bittensor as bt
from typing import Dict, Any, List
from starlette.types import Send

async def chain_forward(
    self,
    query: str,
    files: List,
    extra_info: Dict[str, Any],
    init_time: float,
    timeout_threshold: float,
    chain,
    chain_formatter,
    send: Send,
):
    """
    Helper function for streaming LangChain chain outputs
    """
    buffer = []
    temp_completion = ""
    timeout_reached = False

    try:
        for token in chain.stream(chain_formatter):
            buffer.append(token)

            if time.time() - init_time > timeout_threshold:
                bt.logging.debug(f"⏰ Timeout reached, stopping streaming")
                timeout_reached = True
                break

            if len(buffer) >= self.config.neuron.streaming_batch_size:
                joined_buffer = "".join(buffer)
                temp_completion += joined_buffer
                bt.logging.debug(f"Streamed tokens batch: {len(joined_buffer)} chars")

                await send(
                    {
                        "type": "http.response.body",
                        "body": joined_buffer.encode("utf-8"),
                        "more_body": True,
                    }
                )
                buffer = []

        if buffer and not timeout_reached:
            joined_buffer = "".join(buffer)
            temp_completion += joined_buffer

            await send(
                {
                    "type": "http.response.body",
                    "body": joined_buffer.encode("utf-8"),
                    "more_body": False,
                }
            )

    except Exception as e:
        bt.logging.error(f"Error in chain_forward: {e}")
        if self.config.neuron.stop_on_forward_exception:
            self.should_exit = True

async def string_forward(result_string: str, send: Send):
    """
    Helper function for streaming a simple string
    """
    try:
        await send(
            {
                "type": "http.response.body",
                "body": result_string.encode("utf-8"),
                "more_body": False,
            }
        )
    except Exception as e:
        bt.logging.error(f"Error in string_forward: {e}")
