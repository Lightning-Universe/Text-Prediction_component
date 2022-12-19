import concurrent.futures
import os
import sys
from pathlib import Path
from subprocess import Popen
from time import time
from typing import Optional
from uuid import uuid4

import lightning as L
from fsspec.implementations.local import LocalFileSystem
from lightning.app.storage import Drive
from lightning.app.storage.path import _filesystem
from lightning.app.utilities.cloud import is_running_in_cloud
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class DriveTensorBoardLogger(L.pytorch.loggers.TensorBoardLogger):
    def __init__(self, *args, drive: Drive, refresh_time: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestamp = None
        self.drive = drive
        self.refresh_time = refresh_time

    @rank_zero_only
    def log_metrics(self, metrics, step) -> None:
        super().log_metrics(metrics, step)
        if self.timestamp is None:
            self._upload_to_storage()
            self.timestamp = time()
        elif (time() - self.timestamp) > self.refresh_time:
            self._upload_to_storage()
            self.timestamp = time()

    def _upload_to_storage(self):
        fs = _filesystem()
        fs.invalidate_cache()

        source_path = Path(self.log_dir).resolve()
        destination_path = self.drive._to_shared_path(
            self.log_dir, component_name=self.drive.component_name
        )

        def _copy(from_path: Path, to_path: Path) -> Optional[Exception]:

            try:
                # NOTE: S3 does not have a concept of directories, so we do not need to create one.
                if isinstance(fs, LocalFileSystem):
                    fs.makedirs(str(to_path.parent), exist_ok=True)

                fs.put(str(from_path), str(to_path), recursive=False)

                # Don't delete tensorboard logs.
                if "events.out.tfevents" not in str(from_path):
                    os.remove(str(from_path))

            except Exception as e:
                # Return the exception so that it can be handled in the main thread
                return e

        src = [file for file in source_path.rglob("*") if file.is_file()]
        dst = [destination_path / file.relative_to(source_path) for file in src]

        with concurrent.futures.ThreadPoolExecutor(4) as executor:
            results = executor.map(_copy, src, dst)

        # Raise the first exception found
        exception = next((e for e in results if isinstance(e, Exception)), None)
        if exception:
            raise exception


class TensorBoardWork(L.app.LightningWork):
    def __init__(self, *args, drive: Drive, **kwargs):
        super().__init__(
            *args,
            parallel=True,
            cloud_build_config=L.BuildConfig(requirements=["tensorboard"]),
            **kwargs,
        )

        self.drive = drive

    def run(self):

        use_localhost = not is_running_in_cloud()

        local_folder = f"./tensorboard_logs/{uuid4()}"

        os.makedirs(local_folder, exist_ok=True)

        # Note: Used tensorboard built-in sync methods but it doesn't seem to work.
        cmd = (
            f"tensorboard --logdir={local_folder} --host {self.host} --port {self.port}"
        )

        if use_localhost:
            # installs tensorboard ONLY in the process it needs to be in
            # necessary because local build configs are not yet supported
            cmd = (
                "GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1 GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1 "
                + sys.executable
                + " -m pip install tensorboard && "
                + cmd
            )

        self._process = Popen(cmd, shell=True, env=os.environ)
        print(f"Running Tensorboard on {self.host}:{self.port}")

        fs = _filesystem()
        root_folder = str(self.drive.drive_root)

        while True:
            fs.invalidate_cache()
            for dir, _, files in fs.walk(root_folder):
                for filepath in files:
                    if "events.out.tfevents" not in filepath:
                        continue
                    source_path = os.path.join(dir, filepath)
                    target_path = os.path.join(dir, filepath).replace(
                        root_folder, local_folder
                    )
                    if use_localhost:
                        parent = Path(target_path).resolve().parent
                        if not parent.exists():
                            parent.mkdir(exist_ok=True, parents=True)
                    fs.get(source_path, str(Path(target_path).resolve()))

    def on_exit(self):
        assert self._process
        self._process.kill()
