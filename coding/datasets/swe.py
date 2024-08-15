import os
import random
import requests

from fastcore.xtras import obj2dict
from swebench.collect.build_dataset import create_instance

from .base import Dataset
from coding.helpers.selector import Selector
from coding.helpers.swebench import Repo as SWERepo


def get_package_stats(package_name: str):
    package_url = f"https://pypi.org/pypi/{package_name}/json"
    package_github = None
    response = requests.get(package_url)
    if response.status_code != 200:
        raise Exception(f"Failed to get package data from URL: {package_url}")
    response = response.json()
    if "info" in response:
        if (
            "Source" in response["info"]["project_urls"]
            and "github" in response["info"]["project_urls"]["Source"]
        ):
            package_github = response["info"]["project_urls"]["Source"]
        elif (
            "Homepage" in response["info"]["project_urls"]
            and "github" in response["info"]["project_urls"]["Homepage"]
        ):
            package_github = response["info"]["project_urls"]["Homepage"]
    if not package_github:
        raise Exception(f"No github link found for package: {package_name}")

    return {
        "name": package_name,
        "url": package_url,
        "github": package_github,
    }


def get_top_pip_packages():
    response = requests.get(
        "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json"
    )
    packages = [row["project"] for row in response.json()["rows"]]
    return packages


class SWEDataset(Dataset):
    name = "swe"

    def __init__(
        self,
        seed=None,
    ):
        if seed is None:
            seed = random.randint(0, 1000)
        self.seed = seed

    def get(self, n=100, selector: Selector = None) -> dict:
        random.seed(self.seed)
        # package_name = selector(get_top_pip_packages())
        package_name = "boto3"
        package_info = get_package_stats(package_name)
        token = os.environ.get("GITHUB_TOKEN", None)
        if not token:
            raise Exception("GITHUB_TOKEN not set")

        repo = SWERepo(
            package_info["github"].split("/")[-2],
            package_info["github"].split("/")[-1],
            token,
        )

        valid_pull = None
        err_count = 0
        for pull in repo.get_all_pulls():
            try:
                if valid_pull or err_count > 5:
                    break
                resolved_issues = repo.extract_resolved_issues(pull)
                setattr(pull, "resolved_issues", resolved_issues)
                if len(resolved_issues) > 0:
                    valid_pull = obj2dict(pull)
            except:
                err_count += 1

        if not valid_pull:
            raise Exception("Could not get a valid SWE pull")
        pull_data = create_instance(repo, valid_pull)
        diff_text = pull_data["patch"]
        return {
            "topic": pull_data["problem_statement"],
            "title": f'{package_info["github"].split("/")[-2]}/{package_info["github"].split("/")[-1]}',
            "content": diff_text,
            "extras": dict(pull_number=pull_data["pull_number"]),
        }

    def search(self, query, selector: Selector = None, **kwargs):
        pass

    def random(self, n=100, selector: Selector = None, **kwargs):
        return self.get(n=100, selector=selector)