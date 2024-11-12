import os
from huggingface_hub import HfApi
from coding.model.schema import Model
from constants import CompetitionParameters, MAX_HUGGING_FACE_BYTES

from huggingface_hub import HfApi, file_exists
from collections import defaultdict


def shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for ptr, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing


class HuggingFaceModelStore():
    """Hugging Face based implementation for storing and retrieving a model."""

    @classmethod
    def assert_access_token_exists(cls) -> str:
        """Asserts that the access token exists."""
        if not os.getenv("HF_ACCESS_TOKEN"):
            raise ValueError("No Hugging Face access token found to write to the hub.")
        return os.getenv("HF_ACCESS_TOKEN")

    async def upload_model(
        self,
        model: Model,
        local_path: str,
    ) -> Model:
        """Uploads a trained model to Hugging Face."""
        token = HuggingFaceModelStore.assert_access_token_exists()
        api = HfApi(token=token)
        api.create_repo(
            repo_id=model.model_name,
            exist_ok=True,
            private=True,
        )

        # upload model.local_repo_dir to Hugging Face
        commit_info = api.upload_folder(
            repo_id=model.model_name,
            folder_path=local_path,
            commit_message="Upload model",
            repo_type="model",
        )

        model_id_with_commit = Model(
            model_name=model.model_name,
            chat_template=model.chat_template,
            hash=model.hash,
            commit=commit_info.oid,
            competition_id=model.competition_id,
        )

        return model_id_with_commit
