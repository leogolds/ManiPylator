import dagger
from dagger import dag, function, object_type
from typing import Annotated


@object_type
class Manipylator:
    @function
    def base_container(
        self,
        cuda_version: Annotated[str, "CUDA version to use"] = "12.4",
        python_version: Annotated[str, "Python version to use"] = "3.11",
        registry_username: Annotated[
            str, "Container registry username (optional)"
        ] = "",
        registry_password: Annotated[
            dagger.Secret | None, "Container registry password (optional)"
        ] = None,
    ) -> dagger.Container:
        """Returns a base container with PyTorch CUDA runtime similar to the lab Dockerfile"""
        container = dag.container()

        # Authenticate with registry if credentials are provided
        if registry_username and registry_password:
            container = container.with_registry_auth(
                address="docker.io",
                username=registry_username,
                secret=registry_password,
            )

        return (
            container.from_(f"pytorch/pytorch:2.5.1-cuda{cuda_version}-cudnn9-devel")
            .with_env_variable("DEBIAN_FRONTEND", "noninteractive")
            .with_env_variable("NVIDIA_DRIVER_CAPABILITIES", "all")
            .with_exec(["apt-get", "update"])
            .with_exec(
                [
                    "apt-get",
                    "install",
                    "-y",
                    "--no-install-recommends",
                    "tmux",
                    "git",
                    "curl",
                    "wget",
                    "bash-completion",
                    "libgl1",
                    "libgl1-mesa-glx",
                    "libegl-dev",
                    "libegl1",
                    "libxrender1",
                    "libglib2.0-0",
                    "ffmpeg",
                    "libgtk2.0-dev",
                    "pkg-config",
                    "libvulkan-dev",
                    "libgles2",
                    "libglvnd0",
                    "libglx0",
                ]
            )
            .with_exec(["apt", "clean"])
            .with_exec(["rm", "-rf", "/var/lib/apt/lists/*"])
            .with_mounted_cache("/cache/pip", dag.cache_volume("pip-cache"))
            .with_env_variable("PIP_CACHE_DIR", "/cache/pip")
            .with_workdir("/workspace")
        )

    @function
    async def dev_container(
        self,
        source: Annotated[dagger.Directory, "Source directory"],
        registry_username: Annotated[
            str, "Container registry username (optional)"
        ] = "",
        registry_password: Annotated[
            dagger.Secret | None, "Container registry password (optional)"
        ] = None,
    ) -> dagger.Container:
        """Create a development container with all dependencies"""
        container = (
            self.base_container(
                registry_username=registry_username,
                registry_password=registry_password,
            )
            .with_exec(["pip", "install", "open3d"])
            .with_exec(["pip", "install", "PyOpenGL==3.1.5"])
        )

        # Extract OMPL 1.7 cp311 wheel from zip using disposable container
        ompl_zip = dag.http(
            "https://github.com/ompl/ompl/releases/download/1.7.0/wheels-ubuntu-latest-x86_64.zip"
        )
        ompl_wheel = (
            dag.container()
            .from_("alpine:latest")
            .with_exec(["apk", "add", "unzip"])
            .with_mounted_file("/tmp/ompl.zip", ompl_zip)
            .with_exec(["unzip", "/tmp/ompl.zip", "-d", "/tmp/"])
            .with_exec(["find", "/tmp/", "-name", "*cp311*.whl", "-type", "f"])
            .file("/tmp/ompl-1.7.0-cp311-cp311-manylinux_2_28_x86_64.whl")
        )
        ompl_file_name = await ompl_wheel.name()

        # Install OMPL first
        container = (
            container.with_mounted_file(f"/tmp/{ompl_file_name}", ompl_wheel)
            .with_exec(["pip", "install", f"/tmp/{ompl_file_name}"])
            .without_mount(f"tmp/{ompl_file_name}")
        )

        # Install Genesis from current pinned commit
        genesis_repo = (
            dag.git("https://github.com/Genesis-Embodied-AI/Genesis.git")
            .commit("6479cf646de475206e85c9050dd283cae91c43d0")
            .tree()
        )
        container = (
            container.with_mounted_directory("/genesis", genesis_repo)
            .with_workdir("/genesis")
            .with_exec(["pip", "install", "."])
            .without_mount("/genesis")
            .with_workdir("/workspace")
        )

        # # Install development tools
        # container = container.with_exec(
        #     [
        #         "pip",
        #         "install",
        #         "--no-cache-dir",
        #         "jupyterlab",
        #     ]
        # )

        return container

    @function
    async def build_package(
        self,
        source: Annotated[dagger.Directory, "Source directory"],
    ) -> dagger.Directory:
        """Build the Python package wheel"""
        container = self.with_dependencies(source)

        # Install build tools
        container = container.with_exec(
            ["pip", "install", "--no-cache-dir", "build", "twine"]
        )

        # Build the package
        container = container.with_exec(["python", "-m", "build"])

        return container.directory("/workspace/src/dist")

    @function
    async def run_jupyter(
        self,
        source: Annotated[dagger.Directory, "Source directory"],
        port: Annotated[int, "Port to expose Jupyter on"] = 8888,
    ) -> dagger.Service:
        """Start a Jupyter Lab server"""
        container = self.dev_container(source)

        return (
            container.with_exposed_port(port)
            .with_exec(
                [
                    "jupyter",
                    "lab",
                    "--ip=0.0.0.0",
                    f"--port={port}",
                    "--no-browser",
                    "--allow-root",
                    "--NotebookApp.token=''",
                    "--NotebookApp.password=''",
                ]
            )
            .as_service()
        )

    @function
    async def run_app(
        self,
        source: Annotated[dagger.Directory, "Source directory"],
        port: Annotated[int, "Port to expose the app on"] = 5006,
    ) -> dagger.Service:
        """Run the ManiPylator application"""
        container = self.dev_container(source)

        return (
            container.with_exposed_port(port)
            .with_exec(
                [
                    "python",
                    "-m",
                    "manipylator.app",
                    "--port",
                    str(port),
                    "--allow-websocket-origin=*",
                ]
            )
            .as_service()
        )

    @function
    async def container_echo(self, string_arg: str) -> dagger.Container:
        """Returns a container that echoes whatever string argument is provided"""
        return dag.container().from_("alpine:latest").with_exec(["echo", string_arg])

    @function
    async def grep_dir(self, directory_arg: dagger.Directory, pattern: str) -> str:
        """Returns lines that match a pattern in the files of the provided Directory"""
        return await (
            dag.container()
            .from_("alpine:latest")
            .with_mounted_directory("/mnt", directory_arg)
            .with_workdir("/mnt")
            .with_exec(["grep", "-R", pattern, "."])
            .stdout()
        )
