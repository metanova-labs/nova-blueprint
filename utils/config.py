from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import bittensor as bt
from dotenv import load_dotenv

load_dotenv()


class Config(BaseSettings):
    """
    Environment variable settings from .env.example.
    Only includes variables defined in the root .env.example file.
    """

    # Required for validator (from .env.example)
    bt_wallet_cold: str = Field(
        alias="BT_WALLET_COLD", description="Bittensor wallet coldkey name"
    )

    bt_wallet_hot: str = Field(
        alias="BT_WALLET_HOT", description="Bittensor wallet hotkey name"
    )

    subtensor_network: str = Field(
        default="finney", alias="SUBTENSOR_NETWORK", description="Subtensor network URL"
    )

    github_token: Optional[str] = Field(
        default=None, alias="GITHUB_TOKEN", description="GitHub personal access token"
    )

    backend_api_url: str = Field(alias="BACKEND_API_URL", description="Backend API URL")

    backend_api_token: str = Field(
        alias="BACKEND_API_TOKEN", description="Backend API Bearer token"
    )

    # Optional overrides (from .env.example)
    bt_wallets_dir: Optional[str] = Field(
        default=None,
        alias="BT_WALLETS_DIR",
        description="Bittensor wallets directory path",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra env vars

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_and_warn()

    def _validate_and_warn(self):
        """Warn if critical settings are missing."""
        if not self.bt_wallet_cold:
            bt.logging.warning("BT_WALLET_COLD not set; Bittensor operations may fail.")
        if not self.bt_wallet_hot:
            bt.logging.warning("BT_WALLET_HOT not set; Bittensor operations may fail.")
        if not self.backend_api_url:
            bt.logging.warning("BACKEND_API_URL not set; API calls will fail.")
        if not self.backend_api_token:
            bt.logging.warning("BACKEND_API_TOKEN not set; API calls will fail.")
        if not self.github_token:
            bt.logging.warning("GITHUB_TOKEN not set; GitHub operations may fail.")


settings = Config()
