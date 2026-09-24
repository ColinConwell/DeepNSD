import os
import re
import sys
import zipfile
import configparser

# NSD pycortex database (Google Drive)
PYCORTEX_DB_GDRIVE_URL = (
    "https://drive.google.com/file/d/1oMoOGjlrOpkphKPCC4XqUPaT0nu59kPC/view?usp=drive_link"
)

PYCORTEX_DB_ZIP_NAME = "pycortex_db_NSD.zip"

_GDRIVE_FILE_ID = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")


def default_download_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "pycortex"))


def default_pycortex_config_path():
    """User options.cfg path, without importing pycortex (import reads this file)."""
    try:
        import appdirs
        userdir = appdirs.user_data_dir("pycortex", "JamesGao")
    except ImportError:
        userdir = os.path.join(os.path.expanduser("~"), ".config", "pycortex")
    return os.path.join(userdir, "options.cfg")


def gdrive_file_id(url):
    match = _GDRIVE_FILE_ID.search(url)
    if not match:
        raise ValueError(f"Could not parse a Google Drive file id from {url}")
    return match.group(1)


def download_gdrive_file(url, output_path, force=False):
    """Download a Google Drive file. Reuse output_path when it is already a valid zip."""
    import gdown

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not force and os.path.isfile(output_path):
        try:
            check_zip(output_path)
            print(f"Using existing archive: {output_path}")
            return output_path
        except (zipfile.BadZipFile, FileNotFoundError, OSError) as exc:
            print(f"Replacing invalid archive ({exc}): {output_path}")
            os.remove(output_path)

    print(f"Downloading {url}")
    downloaded = gdown.download(
        id=gdrive_file_id(url),
        output=output_path,
        quiet=False,
        resume=True,
    )
    if not downloaded or not os.path.isfile(output_path):
        raise RuntimeError(f"Google Drive download failed: {url}")
    return os.path.abspath(downloaded)


def check_zip(zip_path):
    """Confirm zip_path is a non-empty archive and that every member CRC matches."""
    zip_path = os.path.abspath(zip_path)
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(zip_path)
    if os.path.getsize(zip_path) == 0:
        raise zipfile.BadZipFile(f"Downloaded file is empty: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise zipfile.BadZipFile(f"Downloaded file is not a zip archive: {zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"Corrupt zip member {bad_member} in {zip_path}")
        names = archive.namelist()

    if not names:
        raise zipfile.BadZipFile(f"Zip archive is empty: {zip_path}")

    print(f"Zip OK: {zip_path} ({len(names)} members, {os.path.getsize(zip_path)} bytes)")
    return names


def _is_within(path, root):
    path = os.path.abspath(path)
    root = os.path.abspath(root)
    return path == root or path.startswith(root + os.sep)


def _extract_is_complete(zip_path, dest_dir):
    dest_dir = os.path.abspath(dest_dir)
    if not os.path.isdir(dest_dir):
        return False
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            target = os.path.abspath(os.path.join(dest_dir, info.filename))
            if not _is_within(target, dest_dir):
                raise ValueError(f"Unsafe path in zip: {info.filename}")
            if info.is_dir():
                if not os.path.isdir(target):
                    return False
            elif not os.path.isfile(target):
                return False
            
    return True # archive is complete


def unzip_archive(zip_path, dest_dir, force=False):
    zip_path = os.path.abspath(zip_path)
    dest_dir = os.path.abspath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    if not force and _extract_is_complete(zip_path, dest_dir):
        print(f"Archive already extracted: {dest_dir}")
        return dest_dir

    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            target = os.path.abspath(os.path.join(dest_dir, info.filename))
            if not _is_within(target, dest_dir):
                raise ValueError(f"Unsafe path in zip: {info.filename}")
        print(f"Extracting {zip_path} -> {dest_dir}")
        archive.extractall(dest_dir)

    return dest_dir


def find_filestore(root):
    """Return the directory that contains pycortex subject folders."""
    root = os.path.abspath(root)
    parents = set()
    for dirpath, dirnames, _ in os.walk(root):
        has_surfaces = os.path.isdir(os.path.join(dirpath, "surfaces"))
        has_transforms = os.path.isdir(os.path.join(dirpath, "transforms"))
        if has_surfaces and has_transforms:
            parents.add(os.path.dirname(dirpath))
            dirnames[:] = []

    if not parents:
        raise FileNotFoundError(f"No pycortex subject found under {root}")
    if len(parents) == 1:
        return parents.pop()
    return sorted(parents, key=lambda path: path.count(os.sep))[0]


def download_check_unzip_pycortex_db(download_dir=None, gdrive_url=PYCORTEX_DB_GDRIVE_URL, force=False):
    download_dir = os.path.abspath(download_dir or default_download_dir())
    os.makedirs(download_dir, exist_ok=True)
    zip_path = os.path.join(download_dir, PYCORTEX_DB_ZIP_NAME)
    extract_dir = os.path.join(download_dir, "pycortex_db_NSD")

    zip_path = download_gdrive_file(gdrive_url, zip_path, force=force)
    check_zip(zip_path)
    extract_dir = unzip_archive(zip_path, extract_dir, force=force)
    db_path = find_filestore(extract_dir)
    print(f"Pycortex filestore: {db_path}")
    return db_path


def bind_pycortex_filestore(db_path):
    """Point an already-imported pycortex session at db_path."""
    db_path = os.path.abspath(db_path)
    options = sys.modules.get("cortex.options")
    if options is not None and options.config.has_section("basic"):
        options.config.set("basic", "filestore", db_path)

    database = sys.modules.get("cortex.database")
    if database is None:
        return
    database.default_filestore = db_path
    if getattr(database, "db", None) is not None:
        database.db.filestore = db_path
        database.db._subjects = None


def setup_pycortex_NSD_config(pycortex_config_path, pycortex_db_NSD_path):
    config_fn = pycortex_config_path
    if os.path.isdir(pycortex_config_path):
        config_fn = os.path.join(pycortex_config_path, "options.cfg")

    db_path = os.path.abspath(pycortex_db_NSD_path)
    os.makedirs(os.path.dirname(os.path.abspath(config_fn)), exist_ok=True)

    if os.path.isfile(config_fn):
        with open(config_fn) as handle:
            lines = handle.readlines()

        in_basic = False
        found = False
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                if in_basic and not found:
                    new_lines.append(f"filestore = {db_path}\n")
                    found = True
                in_basic = stripped.lower() == "[basic]"
            if in_basic and stripped.lower().startswith("filestore"):
                new_lines.append(f"filestore = {db_path}\n")
                found = True
                continue
            new_lines.append(line)

        if not found:
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] = new_lines[-1] + "\n"
            new_lines.append("\n[basic]\n" if not in_basic else "")
            new_lines.append(f"filestore = {db_path}\n")

        with open(config_fn, "w") as handle:
            handle.writelines(line for line in new_lines if line != "")
    else:
        config = configparser.ConfigParser()
        config["basic"] = {"filestore": db_path}
        with open(config_fn, "w") as handle:
            config.write(handle)

    bind_pycortex_filestore(db_path)
    print(f"Pycortex config file updated with new 'filestore' path: {db_path}")
    return config_fn


def main(pycortex_config_path=None, pycortex_db_NSD_path=None, download_dir=None,
         gdrive_url=PYCORTEX_DB_GDRIVE_URL, force=False):
    """Download the NSD pycortex database if needed and point pycortex at it.

    Every argument is optional. Omitted paths use the user pycortex config and
    the default download directory next to this script.
    """
    db_path = pycortex_db_NSD_path
    if force or not db_path or not os.path.isdir(db_path):
        db_path = download_check_unzip_pycortex_db(
            download_dir=download_dir,
            gdrive_url=gdrive_url,
            force=force,
        )
    else:
        db_path = os.path.abspath(db_path)
        print(f"Using existing pycortex filestore: {db_path}")

    config_path = pycortex_config_path or default_pycortex_config_path()
    setup_pycortex_NSD_config(config_path, db_path)
    return config_path, db_path


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Download the NSD pycortex database and point pycortex at it.")
    parser.add_argument("--pycortex_config_path", type=str, default=None)
    parser.add_argument("--pycortex_db_NSD_path", type=str, default=None)
    parser.add_argument("--download_dir", type=str, default=None)
    parser.add_argument("--gdrive_url", type=str, default=PYCORTEX_DB_GDRIVE_URL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
