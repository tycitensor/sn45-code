import os
import ast
import json
import docker
import tempfile
import threading
from pathlib import Path

from coding.constants import COMPETITION_ID
from ..helpers.git import GitRepo

def exec_container_with_timeout(container, command, timeout):
    """
    Executes a command in a Docker container with a timeout.
    
    Args:
        container: The Docker container object.
        command: The command to execute.
        timeout: Timeout in seconds.

    Returns:
        Tuple of exec result and logs.

    Raises:
        TimeoutError: If the command takes longer than the timeout.
    """
    exec_result = None
    logs = None
    exception = None

    def target():
        nonlocal exec_result, logs, exception
        try:
            exec_result, logs = container.exec_run(command)
        except Exception as e:
            exception = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError(f"The command '{command}' exceeded the timeout of {timeout} seconds.")
    if exception:
        raise exception
    return exec_result, logs

def build_docker_container(logic_files: dict, hotkey: str, repo_files: dict) -> str:
    """
    Builds a Docker container for evaluating model logic.

    Args:
        logic_files (dict): Dictionary mapping filenames to file contents
        hotkey (str): Unique identifier for the logic
        repo_files (dict): Dictionary mapping filenames to file contents to copy to repo
        repo_path (str): Path to copy repo files to

    Returns:
        str: ID of the built container
    """
    # Initialize Docker client
    client = docker.from_env()

    # Create temporary directory to store files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write logic files to temp directory
        for filename, content in logic_files.items():
            file_path = os.path.join(temp_dir, filename)
            # Create all parent directories
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Create the file and write content
            with open(file_path, "w", encoding="latin-1") as f:
                f.write(content)

        # Write repo files to repo path
        for filename, content in repo_files.items():
            file_path = os.path.join(temp_dir, "repo", filename)
            # Create all parent directories
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Create the file and write content
            with open(file_path, "w", encoding="latin-1") as f:
                f.write(content)

        # Copy Dockerfile and server files
        swe_server_path = Path(__file__).parent / "swe-server"
        for item in swe_server_path.glob("*"):
            if item.is_file():
                dest_path = os.path.join(temp_dir, item.name)
                with open(item, "rb") as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())
            elif item.is_dir():
                dest_dir = os.path.join(temp_dir, item.name)
                os.system(f"cp -r {item} {dest_dir}")

        # Build the container
        try:
            image, logs = client.images.build(
                path=temp_dir, tag=f"swe-logic-{str(hotkey)}-{COMPETITION_ID}".lower(), rm=True
            )
            return image.id

        except docker.errors.BuildError as e:
            print(f"Error building container: {str(e)}")
            raise
        except docker.errors.APIError as e:
            print(f"Docker API error: {str(e)}")
            raise

def run_docker_container(
    image_id: str, repo: GitRepo, hotkey: str, issue_description: str
) -> dict:
    """
    Runs a Docker container for evaluating model logic.

    Args:
        image_id (str): ID of the Docker image to run
        repo (GitRepo): Git repository object containing code to evaluate
        hotkey (str): Unique identifier for the logic
        issue_description (str): Description of the issue to fix

    Returns:
        dict: The patch output from the container
    """
    # Initialize Docker client
    client = docker.from_env()

    container_name = f"swe-logic-{str(hotkey)}-{COMPETITION_ID}".lower()
    
    try:
        # Remove any existing container with the same name
        try:
            existing = client.containers.get(container_name)
            existing.remove(force=True)
        except docker.errors.NotFound:
            pass

        container = client.containers.create(
            image=image_id,
            name=container_name,
            detach=True,
            ports={"3000/tcp": 3000},
            extra_hosts={"host.docker.internal": "host-gateway"},
            environment={"HOST_IP": os.getenv("HOST_IP", "localhost"), "ISSUE_DESCRIPTION": issue_description},
            # environment={"HOST_IP": "host.docker.internal"},
            # auto_remove=True  # Container will be automatically removed when stopped
        )

        # Start the container
        container.start()
        logs = container.logs().decode('utf-8')

        # Wait for container to finish and get logs
        result = container.wait()
        logs = container.logs().decode('utf-8')
        print("===== CONTAINER LOGS =====")
        print(logs)
        print("===== CONTAINER LOGS =====")
        # Parse the patch from the logs
        patch_line = next(line for line in reversed(logs.split('\n')) if line.startswith('Patch:'))
        try:
            # First try parsing as JSON
            patch_dict = json.loads(patch_line.replace('Patch:', '').strip())
        except json.JSONDecodeError:
            # Fall back to safely evaluating as literal Python dict
            patch_dict = ast.literal_eval(patch_line.replace('Patch:', '').strip())

        # Cleanup container
        try:
            container.stop(timeout=1)
            container.remove(force=True)
        except:
            pass

        return patch_dict

    except docker.errors.APIError as e:
        print(f"Docker API error: {str(e)}")
        raise
    

def run_docker_container_from_base(
    image_id: str, repo: GitRepo, hotkey: str, issue_description: str, logic_files: dict
) -> dict:
    """
    Runs a Docker container for evaluating model logic.

    Args:
        image_id (str): ID of the Docker image to run
        repo (GitRepo): Git repository object containing code to evaluate
        hotkey (str): Unique identifier for the logic
        issue_description (str): Description of the issue to fix

    Returns:
        dict: The patch output from the container
    """
    # Initialize Docker client
    client = docker.from_env()
    container_name = f"swe-logic-{str(hotkey)}-{COMPETITION_ID}".lower()
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write logic files to temp directory
        for filename, content in logic_files.items():
            file_path = os.path.join(temp_dir, filename)
            # Create all parent directories
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Create the file and write content
            with open(file_path, "w", encoding="latin-1") as f:
                f.write(content)

        # Write repo files to repo path
        for filename, content in repo.files.items():
            file_path = os.path.join(temp_dir, "repo", filename)
            # Create all parent directories
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Create the file and write content
            with open(file_path, "w", encoding="latin-1") as f:
                f.write(content)

        # Copy Dockerfile and server files
        swe_server_path = Path(__file__).parent / "swe-server"
        for item in swe_server_path.glob("*"):
            if item.is_file():
                dest_path = os.path.join(temp_dir, item.name)
                with open(item, "rb") as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())
            elif item.is_dir():
                dest_dir = os.path.join(temp_dir, item.name)
                os.system(f"cp -r {item} {dest_dir}")

        try:
            # Remove any existing container with the same name
            try:
                existing = client.containers.get(container_name)
                existing.remove(force=True)
            except docker.errors.NotFound:
                pass

            container = client.containers.create(
                image="brokespace/swe-server:latest",
                name=container_name,
                detach=True,
                ports={"3000/tcp": 3000},
                extra_hosts={"host.docker.internal": "host-gateway"},
                environment={"HOST_IP": os.getenv("HOST_IP", "localhost"), "ISSUE_DESCRIPTION": issue_description},
                command="sleep infinity"
            )

            # Start the container
            container.start()

            # Copy files from temp_dir into container
            os.system(f"docker cp {temp_dir}/. {container_name}:/app/")
            
            # Execute runner.py in container
            exec_result, logs = exec_container_with_timeout(container, "python3 -u /app/runner.py", 600)
            logs = logs.decode('utf-8')

            # Parse the patch from the logs
            patch_line = next(line for line in reversed(logs.split('\n')) if line.startswith('Patch:'))
            try:
                # First try parsing as JSON
                patch_dict = json.loads(patch_line.replace('Patch:', '').strip())
            except json.JSONDecodeError:
                # Fall back to safely evaluating as literal Python dict
                patch_dict = ast.literal_eval(patch_line.replace('Patch:', '').strip())

            # Cleanup container
            try:
                container.stop(timeout=1)
                container.remove(force=True)
            except:
                pass

            return patch_dict

        except docker.errors.APIError as e:
            print(f"Docker API error: {str(e)}")
            raise