from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import load_runtime_config, load_yaml_config


class ConfigLoaderTests(unittest.TestCase):
    def test_yaml_loader_flattens_nested_keys_to_env_style(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    kb_answer_provider: azure_openai
                    azure_openai:
                      api_key: test-key
                      endpoint: https://example.openai.azure.com
                      chat_deployment: gpt-4o-mini
                    qdrant:
                      prefer_grpc: false
                    qdrant_document_collection: customer_care_documents_kb
                    documents_manifest_path: cob_mock_kb_large/high_quality_documents/documents_manifest.json
                    """
                ).strip(),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                load_yaml_config(config_path)

                self.assertEqual(os.environ["KB_ANSWER_PROVIDER"], "azure_openai")
                self.assertEqual(os.environ["AZURE_OPENAI_API_KEY"], "test-key")
                self.assertEqual(
                    os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
                    "gpt-4o-mini",
                )
                self.assertEqual(os.environ["QDRANT_PREFER_GRPC"], "false")
                self.assertEqual(
                    os.environ["QDRANT_DOCUMENT_COLLECTION"],
                    "customer_care_documents_kb",
                )
                self.assertEqual(
                    os.environ["DOCUMENTS_MANIFEST_PATH"],
                    "cob_mock_kb_large/high_quality_documents/documents_manifest.json",
                )

    def test_runtime_config_lets_dotenv_override_yaml_but_not_real_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.yml"
            env_path = temp_path / ".env"
            config_path.write_text(
                textwrap.dedent(
                    """
                    kb_answer_provider: gemini
                    azure_openai:
                      api_key: yaml-key
                    """
                ).strip(),
                encoding="utf-8",
            )
            env_path.write_text(
                textwrap.dedent(
                    """
                    KB_ANSWER_PROVIDER=azure_openai
                    AZURE_OPENAI_API_KEY=dotenv-key
                    """
                ).strip(),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"AZURE_OPENAI_API_KEY": "shell-key"},
                clear=True,
            ):
                load_runtime_config(config_path=config_path, env_path=env_path)

                self.assertEqual(os.environ["KB_ANSWER_PROVIDER"], "azure_openai")
                self.assertEqual(os.environ["AZURE_OPENAI_API_KEY"], "shell-key")


if __name__ == "__main__":
    unittest.main()
