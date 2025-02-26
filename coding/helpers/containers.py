import io
import docker
import logging
from docker.errors import BuildError, APIError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class DockerServer:
    """
    Main class to manage local and remote Docker operations.
    It provides two attributes:
      - server.local: For local Docker operations.
      - server.remote: For operations that build locally and run remotely.
    """

    def __init__(self, remote_host_url: str = None, remote_host_registry: str = None):
        self.remote_host_registry = remote_host_registry
        # Initialize local Docker client
        try:
            self._local_client = docker.from_env()
            logging.info("Connected to local Docker daemon.")
        except Exception as e:
            logging.error("Failed to initialize local Docker client: %s", e)
            raise
        if remote_host_url is not None:
            # Initialize remote Docker client
            try:
                self._remote_client = docker.DockerClient(base_url=remote_host_url)
                self._remote_client.ping()
                logging.info("Connected to remote Docker host at %s", remote_host_url)
            except Exception as e:
                logging.error(
                    "Failed to connect to remote Docker host at %s: %s",
                    remote_host_url,
                    e,
                )
                raise

        # Create handler objects for local and remote operations.
        self.local = LocalDockerHandler(self)
        self.remote = RemoteDockerHandler(self) if remote_host_url is not None else None

    def export_image(self, image_tag: str) -> io.BytesIO:
        """
        Exports the image with the given tag as an in-memory tar archive.

        Args:
            image_tag: The tag of the Docker image to export
        """
        logging.info("Exporting image %s to tar archive", image_tag)
        try:
            image = self._local_client.images.get(image_tag)
            image_tar = io.BytesIO()
            # image.save returns a generator; 'named=True' preserves tags.
            for chunk in image.save(named=True):
                image_tar.write(chunk)
            image_tar.seek(0)
            logging.info("Image export complete.")
            return image_tar
        except Exception as e:
            logging.error("Error exporting image: %s", e)
            raise

    def load_image_remote(self, local_image_tag: str):
        """
        Pushes a local image to the remote registry and loads it on the remote Docker daemon.

        Args:
            local_image_tag: The tag of the local Docker image to transfer
        """
        logging.info(f"Transferring image {local_image_tag} to remote host")
        try:
            if (
                self.remote_host_registry is not None
                and self.remote_host_registry not in local_image_tag
            ):
                # Tag image for remote registry
                remote_tag = f"{self.remote_host_registry}/{local_image_tag}"
                self._local_client.images.get(local_image_tag).tag(remote_tag)
            else:
                remote_tag = local_image_tag

            # Push to remote registry
            logging.info(f"Pushing image to remote registry as {remote_tag}")
            for line in self._local_client.images.push(
                remote_tag, stream=True, decode=True
            ):
                if "status" in line:
                    logging.info(line["status"])

            # Pull on remote host
            logging.info(f"Pulling image on remote host {remote_tag}")
            self._remote_client.images.pull(remote_tag)
            logging.info("Successfully transferred image to remote host")

            return self._remote_client.images.get(remote_tag)

        except Exception as e:
            logging.error(f"Error transferring image to remote host: {e}")
            raise

    def upload_image_to_dockerhub(
        self, image: docker.models.images.Image, repository: str, tag: str = "latest"
    ):
        """
        Uploads a Docker image to DockerHub.

        Args:
            image: The Docker image to upload
            repository: The repository name (e.g. 'username/imagename')
            tag: The tag to use for the image (default: 'latest')
        """
        logging.info(
            f"Uploading image to DockerHub repository '{repository}' with tag '{tag}'"
        )
        try:
            # Tag the image with the repository and tag
            image.tag(repository, tag)

            # Push the image to DockerHub
            for line in self._local_client.images.push(
                repository, tag, stream=True, decode=True
            ):
                if "status" in line:
                    logging.info(line["status"])

            logging.info(f"Successfully uploaded image to {repository}:{tag}")
        except Exception as e:
            logging.error(f"Error uploading image to DockerHub: {e}")
            raise


class LocalDockerHandler:
    """
    Handles Docker operations on the local daemon.
    """

    def __init__(self, server: DockerServer):
        self.server = server
        self.client = server._local_client

    def build(
        self,
        path: str,
        tag: str,
        dockerfile: str = "Dockerfile",
        buildargs: dict = None,
        rm: bool = True,
        decode: bool = False,
        push: bool = False,
    ) -> str:
        """
        Build an image locally. First checks if image exists in DockerHub.
        If push=True, pushes the built image to DockerHub.
        """
        logging.info(f"Checking if image '{tag}' exists in DockerHub")
        try:
            # Try to pull the image from DockerHub
            self.client.images.pull(tag)
            logging.info(
                f"Image '{tag}' found in DockerHub, pulling instead of building"
            )
            return self.client.images.get(tag)
        except Exception:
            logging.info(f"Image '{tag}' not found in DockerHub, proceeding with build")

        logging.info("Building image locally with tag '%s' from %s", tag, dockerfile)
        try:
            image, logs = self.client.images.build(
                path=path,
                dockerfile=dockerfile,
                tag=tag,
                buildargs=buildargs,
                rm=rm,
                decode=decode,
            )
            # Optionally log build output.
            for chunk in logs:
                if "stream" in chunk:
                    logging.info(chunk["stream"].strip())
            logging.info("Local build successful: %s", tag)

            if push and tag:
                # Split tag into repository and tag parts
                repository, tag_part = tag.split(":") if ":" in tag else (tag, "latest")
                logging.info(f"Pushing image to DockerHub: {repository}:{tag_part}")
                try:
                    for line in self.client.images.push(
                        repository, tag_part, stream=True, decode=True
                    ):
                        if "status" in line:
                            logging.info(line["status"])
                    logging.info(
                        f"Successfully pushed image to DockerHub: {repository}:{tag_part}"
                    )
                except Exception as e:
                    logging.error(f"Error pushing to DockerHub: {e}")
                    raise

            return tag
        except (BuildError, APIError) as e:
            logging.error("Local build error: %s", e)
            raise

    def run(
        self,
        image: str,
        command: str = None,
        name: str = None,
        ports: dict = None,
        environment: dict = None,
        detach: bool = True,
        **kwargs,
    ):
        """
        Run a container on the local Docker host.
        """
        logging.info("Running container locally from image '%s'", image)
        try:
            container = self.client.containers.run(
                image,
                command=command,
                name=name,
                ports=ports,
                environment=environment,
                detach=detach,
                **kwargs,
            )
            logging.info("Container started locally with ID %s", container.short_id)
            return container
        except Exception as e:
            logging.error("Error running container locally: %s", e)
            raise


class RemoteDockerHandler:
    """
    Handles operations intended for the remote Docker daemon.
    The remote build method builds the image locally then transfers it remotely.
    """

    def __init__(self, server: DockerServer):
        self.server = server
        self.client = server._remote_client

    def build(
        self,
        path: str,
        tag: str,
        dockerfile: str = "Dockerfile",
        buildargs: dict = None,
        rm: bool = True,
        decode: bool = False,
        push: bool = False,
    ) -> str:
        """
        Builds an image locally, exports it, and then loads it into the remote Docker daemon.
        """
        logging.info(
            "Building image for remote host with tag '%s' from %s", tag, dockerfile
        )
        try:
            # Build locally using the local handler.
            image = self.server.local.build(
                path, tag, dockerfile, buildargs, rm, decode, push
            )
            # Load the image into the remote Docker daemon.
            self.server.load_image_remote(image)
            logging.info("Remote build successful, image '%s' transferred.", tag)
            return tag
        except Exception as e:
            logging.error("Remote build error: %s", e)
            raise

    def run(
        self,
        image: str,
        command: str = None,
        name: str = None,
        ports: dict = None,
        environment: dict = None,
        detach: bool = True,
        **kwargs,
    ):
        """
        Run a container on the remote Docker host.
        If the image doesn't exist remotely, attempt to get it from the local daemon.
        """
        logging.info("Running container on remote host from image '%s'", image)
        try:
            try:
                # Try to get the image from remote
                self.client.images.get(image)
            except docker.errors.ImageNotFound:
                # If not found remotely, try to get it locally and transfer
                logging.info(
                    "Image not found on remote host, attempting to transfer from local"
                )
                try:
                    local_image = self.server.local.client.images.get(image)

                    self.server.load_image_remote(local_image.tags[0])
                except Exception as e:
                    logging.error(
                        "Error transferring image from local to remote: %s", e
                    )
                    raise

            container = self.client.containers.run(
                image,
                command=command,
                name=name,
                ports=ports,
                environment=environment,
                detach=detach,
                **kwargs,
            )
            logging.info(
                "Container started on remote host with ID %s", container.short_id
            )
            return container
        except Exception as e:
            logging.error("Error running container on remote host: %s", e)
            raise


if __name__ == "__main__":
    import os

    REMOTE_DOCKER_URL = os.getenv(
        "REMOTE_DOCKER_HOST"
    )  # Update with your remote host details

    # Instantiate the DockerServer. This sets up both local and remote handlers.
    server = DockerServer(remote_host_url=REMOTE_DOCKER_URL)

    try:
        # ---------------------------
        # Build an image locally only:
        # ---------------------------
        local_image = server.local.build(
            path="./path/to/context",
            dockerfile="Dockerfile",
            tag="my-app:local",
            buildargs={"MY_ARG": "local"},
        )
        logging.info("Local image built: %s", local_image.tags)

        # ---------------------------
        # Build an image for the remote host:
        # (This builds the image locally then transfers it remotely.)
        # ---------------------------
        remote_image = server.remote.build(
            path="./path/to/context",
            dockerfile="Dockerfile",
            tag="my-app:remote",
            buildargs={"MY_ARG": "remote"},
        )
        logging.info("Remote image built and transferred: %s", remote_image.tags)

        # ---------------------------
        # Run a container on the remote host:
        # ---------------------------
        remote_container = server.remote.run(
            image="my-app:remote",
            command="python app.py",
            name="my_app_container",
            ports={"5000/tcp": 5000},
            environment={"ENV_VAR": "production"},
        )
        logging.info("Remote container running with ID: %s", remote_container.short_id)

    except Exception as err:
        logging.error("Operation failed: %s", err)
