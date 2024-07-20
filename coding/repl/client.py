import httpx
from typing import List
from pydantic import BaseModel
from coding.schemas import File

class PackageInfo(BaseModel):
    name: str
    session_id: str = ""

class ScriptRequest(BaseModel):
    session_id: str
    script: str

class FullProcessRequest(BaseModel):
    packages: List[str]
    code_files: List[File]
    script: str


class REPLClient:
    def __init__(self, base_url: str = "http://localhost:15000"):
        self.base_url = base_url
        self.client = httpx.Client()

    def install_package(self, package_info: PackageInfo) -> dict:
        response = self.client.post(
            f"{self.base_url}/install", json=package_info.dict()
        )
        response.raise_for_status()
        return response.json()

    def get_package_code(self, package_info: PackageInfo) -> List[File]:
        response = self.client.post(
            f"{self.base_url}/get_code", json=package_info.dict(), timeout=30
        )
        print(response.json())
        response.raise_for_status()
        return [File(**file) for file in response.json()]

    def update_package_code(
        self, package_info: PackageInfo, updated_files: List[File]
    ) -> dict:
        files_data = [file.dict() for file in updated_files]
        response = self.client.post(
            f"{self.base_url}/update_code",
            json={"package_info": package_info.dict(), "updated_files": files_data},
        )
        response.raise_for_status()
        return response.json()

    def run_script(self, script_request: ScriptRequest) -> dict:
        response = self.client.post(
            f"{self.base_url}/run_script", json=script_request.dict()
        )
        response.raise_for_status()
        return response.json()

    def delete_session(self, session_id: str) -> dict:
        response = self.client.delete(
            f"{self.base_url}/delete_session/{session_id}"
        )
        response.raise_for_status()
        return response.json()

    def cleanup_packages(self, hours: int = 24) -> dict:
        response = self.client.post(
            f"{self.base_url}/cleanup", json={"hours": hours}
        )
        response.raise_for_status()
        return response.json()

    def run_and_delete(
        self, package_name: str, updated_files: List[File], script: str
    ) -> dict:
        files_data = [file.dict() for file in updated_files]
        response = self.client.post(
            f"{self.base_url}/run_and_delete",
            json={"package": package_name, "code_files": files_data, "script": script},
            timeout=20
        )
        print(response.json())
        response.raise_for_status()
        return response.json()

    def close(self):
        self.client.close()
