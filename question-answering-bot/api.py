from typing import List, Type

from pydantic import Field
from steamship.agents.functional import FunctionsBasedAgent
from steamship.agents.llms.openai import ChatOpenAI
from steamship.agents.mixins.transports.steamship_widget import SteamshipWidgetTransport
from steamship.agents.schema import Tool
from steamship.agents.service.agent_service import AgentService
from steamship.invocable import Config


class DocumentQAAgentService(AgentService):

    USED_MIXIN_CLASSES = [
        SteamshipWidgetTransport,
    ]
    """USED_MIXIN_CLASSES tells Steamship what additional HTTP endpoints to register on your AgentService."""

    class DocumentQAAgentServiceConfig(Config):
        """Pydantic definition of the user-settable Configuration of this Agent."""

    config: DocumentQAAgentServiceConfig
    """The configuration block that users who create an instance of this agent will provide."""

    tools: List[Tool]
    """The list of Tools that this agent is capable of using."""

    @classmethod
    def config_cls(cls) -> Type[Config]:
        """Return the Configuration class so that Steamship can auto-generate a web UI upon agent creation time."""
        return DocumentQAAgentService.DocumentQAAgentServiceConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tools = []

        # Agent Setup
        # ---------------------

        self.set_default_agent(
            FunctionsBasedAgent(
                tools=self.tools,
                llm=ChatOpenAI(self.client),
            )
        )

        # Communication Transport Setup
        # -----------------------------

        # Support Steamship's web client
        self.add_mixin(
            SteamshipWidgetTransport(
                client=self.client,
                agent_service=self,
            )
        )

