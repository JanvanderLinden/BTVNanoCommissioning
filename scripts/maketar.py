import os
import tarfile


def make_tarfile(output_filename, source_dir, exclude_dirs=[]):
    with tarfile.open(output_filename, "w:gz") as tar:
        for root, dirs, files in os.walk(source_dir):
            dirs[:] = [
                d for d in dirs if d not in exclude_dirs
            ]  # Exclude specified directories
            for file in files:
                file_path = os.path.join(root, file)
                # Never nest the sandbox tarball inside itself: skip the file we
                # are writing plus any stale copy at the top of source_dir.
                # Matched by exact path, not a "*.tar.gz" glob -- that would
                # also drop the JME Summer23/23BPix payloads under data/.
                if os.path.abspath(file_path) in (
                    os.path.abspath(output_filename),
                    os.path.abspath(
                        os.path.join(source_dir, "BTVNanoCommissioning.tar.gz")
                    ),
                ):
                    continue
                tar.add(file_path, arcname=os.path.relpath(file_path, source_dir))


base_dir = "."
jobdirs = [d for d in os.listdir(base_dir) if d.startswith("jobs_")]
make_tarfile(
    "BTVNanoCommissioning.tar.gz",
    base_dir,
    exclude_dirs=["BTVNanoCommissioning.egg-info"] + jobdirs,
)
