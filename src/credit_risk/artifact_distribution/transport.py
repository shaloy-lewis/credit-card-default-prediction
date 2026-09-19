"""Lazy Hugging Face transport isolated from artifact validation logic."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path


class ArtifactTransportError(RuntimeError):
    """Raised when the external artifact service cannot complete an operation."""


class HuggingFaceTransport:
    """Anonymous downloads and explicitly authenticated maintainer publication."""

    def download(
        self,
        *,
        repo_id: str,
        repo_type: str,
        revision: str,
        remote_path: str,
        cache_dir: Path | None,
        offline: bool,
    ) -> Path:
        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError as error:
            raise ArtifactTransportError(
                "Install the project with the 'artifacts' extra to retrieve Hugging Face artifacts."
            ) from error
        try:
            result = hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                filename=remote_path,
                cache_dir=cache_dir,
                local_files_only=offline,
                token=False,
            )
        except Exception as error:
            mode = "offline cache" if offline else "public Hugging Face repository"
            raise ArtifactTransportError(
                f"Unable to retrieve {remote_path!r} from the pinned {mode}: {error}"
            ) from error
        return Path(result)

    def publish(
        self,
        *,
        repo_id: str,
        files: Mapping[str, Path],
        commit_message: str,
    ) -> str:
        try:
            from huggingface_hub import CommitOperationAdd, HfApi
        except ModuleNotFoundError as error:
            raise ArtifactTransportError(
                "Install the project with the 'artifacts' extra to publish Hugging Face artifacts."
            ) from error
        token = os.environ.get("HF_TOKEN")
        try:
            api = HfApi(token=token)
            api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
            info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
            if bool(getattr(info, "private", True)):
                raise ArtifactTransportError("The artifact repository must be public.")
            existing = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
            current_revision = str(info.sha)
            if len(current_revision) != 40 or any(
                character not in "0123456789abcdef" for character in current_revision
            ):
                raise ArtifactTransportError(
                    "Hugging Face did not return a full immutable parent commit SHA."
                )
            conflicts: list[str] = []
            missing: dict[str, Path] = {}
            for remote, local in files.items():
                if remote not in existing:
                    missing[remote] = local
                    continue
                remote_file = self.download(
                    repo_id=repo_id,
                    repo_type="model",
                    revision=current_revision,
                    remote_path=remote,
                    cache_dir=None,
                    offline=False,
                )
                if _sha256(remote_file) != _sha256(local):
                    conflicts.append(remote)
            if conflicts:
                raise ArtifactTransportError(
                    "Refusing to overwrite existing Hugging Face paths: " + ", ".join(conflicts)
                )
            if not missing:
                return current_revision
            operations = [
                CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
                for remote, local in sorted(missing.items())
            ]
            commit = api.create_commit(
                repo_id=repo_id,
                repo_type="model",
                operations=operations,
                commit_message=commit_message,
                parent_commit=current_revision,
            )
        except ArtifactTransportError:
            raise
        except Exception as error:
            raise ArtifactTransportError(
                f"Unable to publish reviewed artifacts: {error}"
            ) from error
        revision = str(commit.oid)
        if len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise ArtifactTransportError("Hugging Face did not return a full immutable commit SHA.")
        return revision


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
