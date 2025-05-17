# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 Broke

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import dotenv
dotenv.load_dotenv('.env')
import time
import typing
import traceback
import importlib
import bittensor as bt

from typing import Awaitable
from rich.console import Console

# Bittensor Miner Template:
import coding

# import base miner class which takes care of most of the boilerplate
from coding.base.miner import BaseMinerNeuron
from coding.utils.config import config as util_config
from coding.miners.swe import miner_process as miner_process_swe
from coding.miners.openrouter import miner_process as miner_process_or
from coding.protocol import StreamCodeSynapse, LogicSynapse, ProvisionKeySynapse, ResultSynapse
from coding.miners.qwen_mistral_miner import miner_init, miner_process

console = Console()

print(f"elo!")

class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        if not config:
            config = util_config(self)
        self.forward_capabilities = [
            {'forward': self.forward, 'blacklist': self.blacklist, 'priority': self.priority},
            {'forward': self.forward_swe, 'blacklist': self.blacklist_swe, 'priority': self.priority_swe},
            {'forward': self.forward_or, 'blacklist': self.blacklist_or, 'priority': self.priority_or},
            {'forward': self.forward_result, 'blacklist': self.blacklist_result, 'priority': self.priority_result},
        ]
        super().__init__(config=config)
        # miner_name = f"coding.miners.{config.miner.name}_miner"  # if config and config.miner else "bitagent.miners.t5_miner"
        # miner_module = importlib.import_module(miner_name)

        self.miner_init = miner_init
        self.miner_process = miner_process

        self.miner_init(self)

    async def forward_result(
        self, synapse: ResultSynapse
    ) -> ResultSynapse:
        bt.logging.info(f"---------> forward_result synapse: >{synapse.result}<")
        if synapse.result == "":
            return synapse
        console.print(synapse.result)
        return synapse

    async def blacklist_result(
        self, synapse: ResultSynapse
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_result(
        self, synapse: ResultSynapse
    ) -> float:
        return await self.priority(synapse)

    async def forward_or(
        self, synapse: ProvisionKeySynapse
    ) -> ProvisionKeySynapse:
        bt.logging.info(f"---------> forward_or")
        return miner_process_or(self, synapse)

    async def blacklist_or(
        self, synapse: ProvisionKeySynapse
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_or(
        self, synapse: ProvisionKeySynapse
    ) -> float:
        return await self.priority(synapse)

    async def priority_swe(
        self, synapse: LogicSynapse
    ) -> float:
        return await self.priority(synapse)

    async def forward_swe(
        self, synapse: LogicSynapse
    ) -> LogicSynapse:
        bt.logging.info(f"---------> forward_swe")
        return miner_process_swe(self, synapse)

    async def blacklist_swe(
        self, synapse: LogicSynapse
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_swe(
        self, synapse: LogicSynapse
    ) -> float:
        return await self.priority(synapse)

    def forward(
        self, synapse: StreamCodeSynapse
    ) -> StreamCodeSynapse:
        """
        Processes the incoming 'Dummy' synapse by performing a predefined operation on the input data.
        This method should be replaced with actual logic relevant to the miner's purpose.

        Args:
            synapse (template.protocol.Dummy): The synapse object containing the 'dummy_input' data.

        Returns:
            template.protocol.Dummy: The synapse object with the 'dummy_output' field set to twice the 'dummy_input' value.

        The 'forward' function is a placeholder and should be overridden with logic that is appropriate for
        the miner's intended operation. This method demonstrates a basic transformation of input data.
        """
        try:
            bt.logging.info( f"------> Response properly passed")
            bt.logging.info( f"------> Response properly passed")
            bt.logging.info( f"------> Response properly passed")
            bt.logging.info( f"------> Response properly passed")
            bt.logging.info( f"------> Response properly passed")
            response = self.miner_process(self, synapse)
            bt.logging.info( f"------> Response properly passed")
        except:
            bt.logging.error(
                "An error occurred while processing the synapse: ",
                traceback.format_exc(),
            )
        return response

    async def blacklist(
        self, synapse: StreamCodeSynapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Dummy): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        try:
            if synapse.dendrite is None or synapse.dendrite.hotkey is None:
                bt.logging.warning("Received a request without a dendrite or hotkey.")
                return True, "Missing dendrite or hotkey"
            if (
                synapse.dendrite.hotkey
                == "5Fy7c6skhxBifdPPEs3TyytxFc7Rq6UdLqysNPZ5AMAUbRQx"
            ):
                return False, "Subnet owner hotkey"
            # TODO(developer): Define how miners should blacklist requests.
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            if (
                not self.config.blacklist.allow_non_registered
                and synapse.dendrite.hotkey not in self.metagraph.hotkeys
            ):
                # Ignore requests from un-registered entities.
                bt.logging.trace(
                    f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Unrecognized hotkey"

            if self.config.blacklist.force_validator_permit:
                # If the config is set to force validator permit, then we should only allow requests from validators.
                if not self.metagraph.validator_permit[uid]:
                    bt.logging.warning(
                        f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                    )
                    return True, "Non-validator hotkey"

            bt.logging.trace(
                f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
            )
            return False, "Hotkey recognized!"
        except:
            return True, "Errored out the blacklist function, blacklisting the hotkey"

    async def priority(
        self, synapse: StreamCodeSynapse
    ) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Dummy): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0
        try:
            caller_uid = self.metagraph.hotkeys.index(
                synapse.dendrite.hotkey
            )  # Get the caller index.
            priority = float(
                self.metagraph.S[caller_uid]
            )  # Return the stake as the priority.
            bt.logging.trace(
                f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
            )
            return priority
        except:
            return 1


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)
