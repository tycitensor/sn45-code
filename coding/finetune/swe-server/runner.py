import os
import submission

swe_instance = submission.SWE()

def run_swe(repo_location, issue_description):
    return swe_instance(repo_location, issue_description)

if __name__ == "__main__":
    repo_location = "/testbed"
    issue_description = os.getenv("ISSUE_DESCRIPTION")
    result = run_swe(repo_location, issue_description)
    print("Patch: ", result.model_dump())