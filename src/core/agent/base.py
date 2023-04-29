import abc
import logging
from abc import ABC
from typing import List, Optional, Dict, Any, Type, Union
from pydantic import Field

from langchain.agents import Tool
from steamship import SteamshipError, Block
from steamship.experimental.transports.chat import ChatMessage
from steamship.invocable import PackageService, post, Config
from core.comms import CommsChannels

from utils import is_valid_uuid


def response_for_exception(e: Optional[Exception]) -> str:
    if e is None:
        return (
            "An unknown error happened. "
            "Please reach out to support@steamship.com or on our discord at https://steamship.com/discord"
        )

    if "usage limit" in f"{e}":
        return (
            "You have reached the introductory limit of Steamship. "
            "Visit https://steamship.com/account/plan to sign up for a plan."
        )

    return f"An error happened while creating a response: {e}"


class BaseAgentConfig(Config):
    """Config object containing parameters to initialize a MyAgent instance."""
    telegram_token: Optional[str] = Field("", description="The secret token for your Telegram bot")


class BaseAgent(PackageService, ABC):
    name: str = "MyAgent"
    config: BaseAgentConfig
    comms: CommsChannels

    @classmethod
    def config_cls(cls) -> Type[Config]:
        return BaseAgentConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comms = CommsChannels(self.client, telegram_token=self.config.telegram_token)

    @abc.abstractmethod
    def get_tools(self) -> List[Tool]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_personality(self) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def is_verbose_logging_enabled(self) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_agent(self):
        raise NotImplementedError()

    def run(self, prompt: str) -> str:
        return self.get_agent().run(input=prompt)

    @post("answer", public=True)
    def answer(self, **kwargs) -> List[Dict[str, Any]]:
        """Endpoint that implements the contract for Steamship embeddable chat widgets.
        This is a PUBLIC endpoint since these webhooks do not pass a token."""
        try:
            input_message = self.comms.webtransport_parse(**kwargs)            
            chain_output = self.run(input_message.text)
            output_messages = self.chain_output_to_chat_messages(input_message, chain_output)
        except SteamshipError as e:
            output_messages = [ChatMessage(
                client=self.client,
                chat_id=kwargs.get("chat_session_id"),
                text=response_for_exception(e)
            )]

        return self.comms.web_transport_send(output_messages)

    @post("info")
    def info(self) -> dict:
        """Endpoint returning information about this bot."""
        return {"telegram": "Hello There!"}

    def chain_output_to_chat_messages(self, inbound_message: ChatMessage, chain_output: Union[str, List[str]]) -> List[ChatMessage]:
        """Transform the output of the Tool/Chain into a list of ChatMessage objects..
        A tool/chain returns a string or list of strings. The string contents contains a sneak-route for mime encoding:
        It is either:
        - A parseable UUID, representing a block containing binary data, or:
        - Text
        This method inspects each string and creates a block of the appropriate type.
        """
        ret = []
        for part_response in chain_output if isinstance(chain_output, list) else [chain_output]:
            if is_valid_uuid(part_response):
                # It's a block containing binary data.
                block = Block.get(self.client, _id=part_response).dict()
                block["who"] = "bot"
                ret.append(
                     ChatMessage(client=self.client, chat_id=inbound_message.get_chat_id(), **block)
                )
            else:
                ret.append(
                    ChatMessage(
                        client=self.client, 
                        chat_id=inbound_message.get_chat_id(), 
                        text=part_response,
                        who="bot"
                    )
                )
        return ret