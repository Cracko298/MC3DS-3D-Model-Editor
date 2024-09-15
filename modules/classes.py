import os
from io import BytesIO
import typing
from pathlib import Path
import zlib
import re

from leveldat import LevelDat

from parser import *


def process_key(key: int, length: int | None = None) -> int:
    if length is not None and key > length - 1:
        raise IndexError("index out of range")
    if key < 0:
        if length is None:
            raise IndexError("cannot use negative indexes with unknown length")
        else:
            key = length + key
            # if it's still negative, it's out of range
            if key < 0:
                raise IndexError("index out of range")
    return key


class BaseParser:
    def __init__(self, stream: BytesIO) -> None:
        self._stream = stream
        self._offset = stream.tell()
        self._reload_data()

    def _reload_data(self) -> None:
        pass

    def _seek(self, position: int) -> None:
        self._stream.seek(self._offset + position)


class Subfile(BaseParser):
    def __init__(self, stream: BytesIO, subfile_size: int) -> None:
        self._size = subfile_size
        super().__init__(stream)

    # it's a property so it's read only
    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self.size

    def _reload_data(self) -> None:
        self._seek(0)
        self._header = parser.SubfileHeader(self._stream)

    @property
    def filler(self) -> bool:
        # some subfiles have a header of all zeroes, and this avoids an error when parsing
        return self._header.magic == 0

    @property
    def raw(self) -> bytes | None:
        if self.filler:
            return None
        self._seek(self._header.size)
        content = self._stream.read(self.size - self._header.size)
        return content


class IterDB:
    def __init__(self, db) -> None:
        self._db = db
        self.__index = 0

    def __next__(self) -> tuple[int, typing.Any]:
        while True:
            if self.__index > len(self._db) - 1:
                raise StopIteration
            new_subfile = self._db[self.__index]
            self.__index += 1
            if not new_subfile.filler:
                break
        return self.__index - 1, new_subfile


class DBFile(BaseParser):
    def _reload_data(self) -> None:
        self._header = parser.FileHeader(self._stream)

    @property
    def something(self) -> tuple[int]:
        return (self._header.something0, self._header.something1)

    @property
    def unknown(self) -> tuple[int]:
        return (self._header.unknown0, self._header.unknown1)

    @property
    def subfile_count(self) -> int:
        return self._header.subfileCount

    @property
    def subfile_size(self) -> int:
        return self._header.subfileSize

    def __len__(self) -> int:
        return self.subfile_count

    def _parse(self, subfile: Subfile):
        return subfile

    def __iter__(self) -> IterDB:
        return IterDB(self)

    def __getitem__(self, key: int) -> bytes | None:
        if key > self.subfile_size - 1:
            raise IndexError("index out of range")
        if key < 0:
            key = self.subfile_size + key
            # if it's still negative, it's out of range
            if key < 0:
                raise IndexError("index out of range")

        self._seek(self.subfile_size * key + parser.FileHeader.size)
        subfile = Subfile(self._stream, self.subfile_size)
        return self._parse(subfile)


class Subchunk:
    def __init__(self, header, compressed: bytes) -> None:
        self._header = header
        self._compressed = compressed

    @property
    def compressed(self) -> bytes:
        return self._compressed

    @property
    def decompressed(self) -> bytes:
        decompress_object = zlib.decompressobj()
        decompressed = decompress_object.decompress(self._compressed)
        compressed_size = len(self._compressed) - len(decompress_object.unused_data)
        assert compressed_size == self._header.compressedSize, (
            f"compressed size {compressed_size:d} "
            f"is not expected size {self._header.compressedSize:d}"
        )
        assert len(decompressed) == self._header.decompressedSize, (
            f"decompressed size {len(decompressed):d} "
            f"is not expected size {self._header.decompressedSize:d}"
        )
        return decompressed


class IterChunk:
    def __init__(self, chunk) -> None:
        self._chunk = chunk
        self.__index = 0

    def __next__(self) -> tuple[int, Subchunk]:
        while True:
            if self.__index > len(self._chunk) - 1:
                raise StopIteration
            new_subchunk = self._chunk[self.__index]
            self.__index += 1
            if new_subchunk is not None:
                break
        return self.__index - 1, new_subchunk


class Chunk:
    def __init__(self, subfile: Subfile) -> None:
        self.__section = 0
        self._subfile = subfile
        self._reload_data()

    def _reload_data(self) -> None:
        self._raw = self._subfile.raw
        self._header = parser.ChunkHeader(self._raw)

    @property
    def unknown0(self) -> int:
        return self._header.unknown0

    @property
    def unknown1(self) -> int:
        return self._header.unknown1

    @property
    def filler(self) -> bool:
        return self._subfile.filler

    @property
    def sections(self) -> int:
        return 0 if self.filler else len(self._header.sections)

    def __len__(self) -> int:
        return self.sections

    def __iter__(self):
        return IterChunk(self)

    def __getitem__(self, key: int) -> Subchunk:
        if self.filler:
            raise ValueError("cannot read filler")
        key = process_key(key, self.sections)

        skipped = 0
        start = self._header.size
        decompress_object = zlib.decompressobj()
        for section in self._header.sections:
            if section.index == key:
                size = section.compressedSize
                break
            elif section.index != -1:
                # skip past the decompressed data
                start += section.compressedSize
        else:
            # doesn't exist
            return None

        subchunk_header = self._header.sections[key]
        should_be = start + parser.SubfileHeader.size
        if subchunk_header.position != should_be:
            raise ValueError("invalid position")
        subchunk = Subchunk(subchunk_header, self._raw[start:])
        return subchunk


class CDBFile(DBFile):
    def _parse(self, subfile: Subfile):
        return Chunk(subfile)


class IterCDBDirectory:
    def __init__(self, cdb_directory):
        self._cdb_directory = cdb_directory
        self._keys = list(sorted(cdb_directory.keys()))

    def __next__(self) -> tuple[int, CDBFile]:
        try:
            new_key = self._keys.pop(0)
        except IndexError:
            raise StopIteration
        return new_key, self._cdb_directory[new_key]


class CDBDirectory:
    cdb_expression = re.compile(R"slt([1-9]\d*)\.cdb")

    def __init__(self, path: str | bytes | os.PathLike) -> None:
        self._path = path
        self._reload_data()

    def __iter__(self) -> IterCDBDirectory:
        return IterCDBDirectory(self)

    def _reload_data(self) -> None:
        cdb_files = filter(lambda path: path.is_file(), self._path.iterdir())
        self._cdb_files = {}
        for cdb_file in cdb_files:
            filename = cdb_file.name
            matched = self.cdb_expression.fullmatch(filename)
            if matched is None:
                continue
            cdb_number = int(matched[1])
            self._cdb_files[cdb_number] = cdb_file

    @property
    def path(self) -> str | bytes | os.PathLike:
        return self._path

    @path.setter
    def path(self, value: str | bytes | os.PathLike) -> None:
        self._path = Path(value)
        self._reload_data()

    def keys(self) -> tuple[int]:
        return tuple(self._cdb_files.keys())

    def get_file(self, key: int) -> Path:
        return self._cdb_files[key]

    def __getitem__(self, key: int):
        cdb_path = self._cdb_files[key]
        return CDBFile(open(cdb_path, "rb"))


class World:
    def __init__(self, path: str | bytes | os.PathLike) -> None:
        self._path = Path(path)
        self._reload_data()

    def _reload_data(self) -> None:
        self._db_path = self._path / "db"
        self._cdb_path = self._db_path / "cdb"
        self._vdb_path = self._db_path / "vdb"
        self.cdb = CDBDirectory(self._cdb_path)

        self._level_path = self._path / "level.dat"
        self._level_old_path = self._path / "level.dat_old"
        with open(self._level_path, "rb") as level_file:
            buffer = level_file.read()
        self.metadata = LevelDat(buffer)
        if self._level_old_path.exists():
            with open(self._level_old_path, "rb") as level_file:
                buffer = level_file.read()
            self.old_metadata = LevelDat(buffer)
        else:
            self.old_metadata = None

    @property
    def path(self) -> Path:
        return self._path

    @path.setter
    def path(self, value: str | bytes | os.PathLike) -> None:
        self._path = Path(value)
        self._reload_data()

    @property
    def name(self) -> str:
        return self.metadata.get("LevelName")
