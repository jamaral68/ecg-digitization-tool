from io import BytesIO
from pathlib import Path
import re
import os
from typing import Literal, get_args, get_type_hints, BinaryIO
import boto3
import botocore


def get_element_list(
    path: str | Path,
    assets: Literal["files", "folders", "both"],
    return_full_path: bool = False,
    recursive: bool = False,
    match_full_path: bool = False,
    sort: bool = True,
    sort_descending: bool = False,
    filt: str | None = None,
) -> list[str]:
    """
    Function for getting the list of files or folders in a directory.

    Parameters
    ----------
    path : str | Path
        Path for the folders/files.
    assets : Literal["files", "folders", "both"]
        Type of assets to search for: 'files', 'folders', or 'both'.
    return_full_path : bool, optional
        Whether to return the complete path or just the basename (Default: False).
    recursive : bool, optional
        Whether to search recursively in subdirectories (Default: False).
    match_full_path : bool, optional
        Whether to apply the filter to the full path or only to the basename
        (Default: False).
    sort : bool, optional
        Whether to sort the returned elements (Default: True).
    sort_descending : bool, optional
        Whether to sort in descending order (Default: False).
    filt : str | None, optional
        Regex filter for filtering the assets name (Default: None).

    Returns
    -------
    list[str]
        List of files or folders in the directory.
    """
    # Validate assets parameter by extracting values from the Literal type hint
    type_hints = get_type_hints(get_element_list)
    valid_assets = get_args(type_hints["assets"])
    if assets not in valid_assets:
        valid_options = ", ".join(f"{opt!r}" for opt in valid_assets)
        raise ValueError(
            f"Invalid value for 'assets': {assets!r}. Must be one of: {valid_options}."
        )

    element_list = []

    for dirpath, dirnames, filenames in os.walk(path):
        assets_to_process = []
        if assets in ("files", "both"):
            assets_to_process.extend(filenames)
        if assets in ("folders", "both"):
            assets_to_process.extend(dirnames)

        element_list.extend([f"{dirpath}/{asset}" for asset in assets_to_process])

        if not recursive:
            break

    if filt is not None:
        if match_full_path:
            element_list = [
                _asset for _asset in element_list if re.search(filt, _asset)
            ]
        else:
            element_list = [
                _asset
                for _asset in element_list
                if re.search(filt, os.path.basename(_asset))
            ]

    if not return_full_path:
        element_list = [os.path.basename(_asset) for _asset in element_list]
    if sort:
        element_list.sort(reverse=sort_descending)
    return element_list


def adjust_remote_path(remote_path: str, bucket_name: str) -> str:
    """
    Convert a remote path in multiple patterns (URI, bucket-prefixed, etc.)
    to a bucket-relative key/prefix.

    Parameters
    ----------
    remote_path : str
        The remote path to adjust (can be URI, bucket-prefixed, etc.)
    bucket_name : str
        The bucket name to remove from the path if present

    Returns
    -------
    str
        The adjusted path relative to the bucket root
    """
    # Remove URI format (gs://bucket/ or s3://bucket/) or bucket prefix (bucket/)
    # Only remove if bucket appears at the start of the path
    uri_pattern = rf"^(gs://|s3://)?{re.escape(bucket_name)}/"
    result = re.sub(uri_pattern, "", remote_path)

    # Remove leading slash if present
    result = re.sub(r"^/", "", result)

    return result


def get_mime_types(suffix: str) -> str | None:
    mime_types = {
        "htm": "text/html",
        "html": "text/html",
        "css": "text/css",
        "js": "text/javascript",
        "json": "application/json",
        "xml": "application/xml",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "ico": "image/x-icon",
        "svg": "image/svg+xml",
        "gif": "image/gif",
        "gpx": "application/gpx+xml",
        "txt": "text/plain",
        "scss": "text/x-scss",
        "eot": "application/vnd.ms-fontobject",
        "pdf": "application/pdf",
        "ttf": "font/ttf",
        "woff": "font/woff",
        "woff2": "font/woff2",
        "mp4": "video/mp4",
        "yaml": "application/x-yaml",
        "inv": "binary/octet-stream",
        "buildinfo": "binary/octet-stream",
        "md": "text/markdown",
        "ipynb": "binary/octet-stream",
        "po": "binary/octet-stream",
        "map": "application/json",
    }
    try:
        return mime_types[suffix[1:]]
    except KeyError:
        return None


class S3Client:
    """
    Simple client to interact with an object storage bucket using AWS S3.
    First initialize the client with the bucket name and the AWS credentials using
    ```aws login``` in the terminal to get the AWS credentials.
    Then you can use the client to upload, download, list, delete files and prefixes.
    """

    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
    ) -> None:
        """
        Initialize the client.

        Parameters
        ----------
        bucket_name: bucket name.\n
        aws_access_key_id: AWS access key id (optional).\n
        aws_secret_access_key: AWS secret access key (optional).\n
        aws_session_token: AWS session token (optional).
        """
        self.bucket_name = bucket_name
        self.s3 = boto3.resource(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )
        self.bucket = self.s3.Bucket(self.bucket_name)

        try:
            self.s3.meta.client.head_bucket(Bucket=self.bucket_name)
        except botocore.exceptions.ClientError as e:
            error_code = int(e.response["Error"]["Code"])
            if error_code == 404:
                print("ERROR - The bucket does not exist.")

    def upload_file(
        self,
        local_source_path: str | Path,
        remote_object_key: str | None = None,
    ) -> None:
        """
        Upload a local file to the bucket.

        Parameters
        ----------
        local_source_path: local file path (absolute or relative).\n
        remote_object_key: destination key in the bucket (accepts URI or partial path).
          If not specified, uses the basename of local_source_path.
        """
        p_local = Path(local_source_path)

        if remote_object_key is None:
            remote_object_key = p_local.name

        remote_object_key = adjust_remote_path(remote_object_key, self.bucket_name)

        suffix = p_local.suffix
        content_type = get_mime_types(suffix)

        if content_type:
            self.bucket.upload_file(
                str(p_local),
                remote_object_key,
                ExtraArgs={"ContentType": content_type},
            )
        else:
            self.bucket.upload_file(str(p_local), remote_object_key)

    def upload_files_prefix(
        self,
        local_source_dir: str | Path,
        remote_prefix: str | None = None,
        filt: str | None = None,
    ) -> None:
        """
        Upload all files from a local directory to a remote prefix.

        Parameters
        ----------
        local_source_dir: local directory path.\n
        remote_prefix: destination prefix in the bucket (accepts URI or partial path).
          If not specified, uses the bucket root.\n
        filt: regex filter applied to file paths (default: no filter).
        """
        local_dir_str = str(local_source_dir)
        if filt is None:
            filt = ""

        file_list = get_element_list(
            path=local_dir_str,
            assets="files",
            return_full_path=True,
            match_full_path=True,
            recursive=True,
            sort=True,
            filt=filt,
        )

        if remote_prefix is None:
            remote_prefix = ""
        remote_prefix_adj = adjust_remote_path(remote_prefix, self.bucket_name)

        for file in file_list:
            p_relative = Path(file).relative_to(local_dir_str)
            self.upload_file(
                local_source_path=file,
                remote_object_key=str(Path(remote_prefix_adj).joinpath(p_relative)),
            )

    def upload_stream(
        self,
        remote_object_key: str,
        fileobj: BinaryIO,
        remote_prefix: str | None = None,
    ) -> None:
        """
        Upload a file-like object to the bucket.

        Parameters
        ----------
        remote_object_key: destination key in the bucket
          (accepts URI or partial path).\n
        fileobj: file-like object (binary stream).\n
        remote_prefix: optional prefix to be prepended to remote_object_key.
        """
        remote_object_key = adjust_remote_path(remote_object_key, self.bucket_name)

        if remote_prefix is None:
            remote_prefix = ""
        remote_prefix_adj = adjust_remote_path(remote_prefix, self.bucket_name)

        final_key = (
            str(Path(remote_prefix_adj, remote_object_key))
            if remote_prefix_adj
            else remote_object_key
        )
        self.bucket.upload_fileobj(Fileobj=fileobj, Key=final_key)

    def list_files(self, remote_prefix: str | None = None) -> list[str]:
        """
        List keys in the bucket (optionally filtered by a prefix).

        Parameters
        ----------
        remote_prefix: prefix filter (accepts URI or partial path). If None, lists all.

        Returns
        -------
        List of keys.
        """
        if remote_prefix is None:
            remote_prefix = ""
        remote_prefix_adj = adjust_remote_path(remote_prefix, self.bucket_name)

        objs = self.bucket.objects.filter(Prefix=remote_prefix_adj).all()
        return [obj.key for obj in objs]

    def delete_files(self, remote_object_keys: list[str]) -> None:
        """
        Delete multiple objects by their keys.

        Parameters
        ----------
        remote_object_keys: list of keys (accepts URI or partial path per entry).
        """
        if not isinstance(remote_object_keys, list):
            raise Exception(
                "remote_object_keys must be a list but "
                f"it was a {type(remote_object_keys)}: {str(remote_object_keys)}"
            )

        delete_keys = {
            "Objects": [
                {"Key": adjust_remote_path(k, self.bucket_name)}
                for k in remote_object_keys
            ]
        }
        self.bucket.delete_objects(Delete=delete_keys)

    def delete_prefix(self, remote_prefix: str) -> list[str]:
        """
        Delete all objects under a prefix (or a single key).

        Parameters
        ----------
        remote_prefix: prefix/key to delete (accepts URI or partial path).

        Returns
        -------
        List of deleted keys.
        """
        remote_prefix_adj = adjust_remote_path(remote_prefix, self.bucket_name)
        resp = self.bucket.objects.filter(Prefix=remote_prefix_adj).delete()
        return [obj["Key"] for obj in resp[0].get("Deleted", [])]

    def download_file(
        self,
        remote_object_key: str,
        local_target_path: str | Path | None = None,
    ) -> None:
        """
        Download an object to a local file.

        Parameters
        ----------
        remote_object_key: key in the bucket (accepts URI or partial path).\n
        local_target_path: local destination path. If not specified, saves to the
          current working directory using the basename of remote_object_key.
        """
        remote_object_key_adj = adjust_remote_path(remote_object_key, self.bucket_name)

        if local_target_path is None:
            local_target_path = Path(remote_object_key_adj).name

        self.bucket.download_file(remote_object_key_adj, str(local_target_path))

    def download_file_bytes(self, remote_object_key: str) -> bytes:
        """
        Download an object and return its content as bytes.

        Parameters
        ----------
        remote_object_key: key in the bucket (accepts URI or partial path).

        Returns
        -------
        File content as bytes.
        """
        remote_object_key_adj = adjust_remote_path(remote_object_key, self.bucket_name)
        with BytesIO() as writter:
            self.bucket.download_fileobj(remote_object_key_adj, writter)
            return writter.getvalue()

    def download_files_bytes(self, remote_object_keys: list[str]) -> list[bytes]:
        """
        Download multiple objects and return their contents as bytes.

        Parameters
        ----------
        remote_object_keys: list of keys (accepts URI or partial path per entry).

        Returns
        -------
        List of file contents as bytes.
        """
        if not isinstance(remote_object_keys, list):
            raise Exception(
                "remote_object_keys must be a list but "
                f"it was a {type(remote_object_keys)}: {str(remote_object_keys)}"
            )

        out: list[bytes] = []
        for key in remote_object_keys:
            key_adj = adjust_remote_path(key, self.bucket_name)
            with BytesIO() as writter:
                self.bucket.download_fileobj(key_adj, writter)
                out.append(writter.getvalue())
        return out

    def download_files_prefix(
        self,
        remote_prefix: str,
        allowed_suffixes: list[str] | None = None,
        local_target_dir: str | Path | None = None,
    ) -> None:
        """
        Download all objects under a remote prefix into the local filesystem.

        Parameters
        ----------
        remote_prefix: prefix to download from (accepts URI or partial path).\n
        allowed_suffixes: list of file suffixes to include (e.g. [".png", ".json"]).
          If None or empty, no filter.\n
        local_target_dir: local directory to save files.
          If None, uses the adjusted remote_prefix as the local base directory
          (mirrors remote structure).
        """
        if allowed_suffixes is None:
            allowed_suffixes = []
        if not isinstance(allowed_suffixes, list):
            raise Exception(
                "allowed_suffixes must be a list but "
                f"it was a {type(allowed_suffixes)}: {str(allowed_suffixes)}"
            )

        remote_prefix_adj = adjust_remote_path(remote_prefix, self.bucket_name)
        file_list = self.list_files(remote_prefix_adj)

        for remote_key in file_list:
            if remote_key.endswith("/"):
                continue

            p = Path(remote_key)

            # Path.relative_to("") raises ValueError. If prefix is empty, use p itself.
            p_relative = p.relative_to(remote_prefix_adj) if remote_prefix_adj else p

            if allowed_suffixes and p.suffix not in allowed_suffixes:
                continue

            base_dir = (
                Path(remote_prefix_adj)
                if local_target_dir is None
                else Path(local_target_dir)
            )
            local_target_path = base_dir.joinpath(p_relative)
            local_target_path.parent.mkdir(parents=True, exist_ok=True)

            self.download_file(
                remote_object_key=remote_key,
                local_target_path=local_target_path,
            )
