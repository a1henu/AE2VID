import pandas as pd
import zipfile
from os.path import splitext
import numpy as np
from .timers import Timer


class FixedSizeEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each containing a fixed number of events.
    """

    def __init__(self, path_to_event_file, num_events=10000, start_index=0):
        print('Will use fixed size event windows with {} events'.format(num_events))
        print('Output frame rate: variable')
        self.iterator = pd.read_csv(path_to_event_file, delim_whitespace=True, header=None,
                                    names=['t', 'x', 'y', 'pol'],
                                    dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.int16},
                                    engine='c',
                                    skiprows=start_index + 1, chunksize=num_events, nrows=None, memory_map=True)

    def __iter__(self):
        return self

    def __next__(self):
        with Timer('Reading event window from file'):
            event_window = self.iterator.__next__().values
        return event_window

class FixedSizeEventNpyReader:
    """
    Reads events from a '.npy' file, and packages the events into
    non-overlapping event windows, each containing a fixed number of events.
    """

    def __init__(self, path: str, num_events: int = 10_000, start_index: int = 0, copy_out: bool = False):
        # mmap_mode='r' 不将文件完全载入内存，适合超大文件
        self.arr = np.load(path, mmap_mode='r')  # 支持普通二维数组，或结构化dtype
        self.n = self.arr.shape[0]
        self.num_events = int(num_events)
        self.i = int(start_index)
        self.copy_out = bool(copy_out)  # 若后续要在GPU里异步用，或担心底层被占用，可 copy

        if self.i < 0 or self.i > self.n:
            raise ValueError(f"start_index {start_index} out of range 0..{self.n}")

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        j = min(self.i + self.num_events, self.n)
        chunk = self.arr[self.i:j]
        self.i = j
        chunk = np.array([chunk['t'], chunk['x'], chunk['y'], chunk['p']]).T
        return chunk.copy() if self.copy_out else chunk


class FixedDurationEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each of a fixed duration.

    **Note**: This reader is much slower than the FixedSizeEventReader.
              The reason is that the latter can use Pandas' very efficient cunk-based reading scheme implemented in C.
    """

    def __init__(self, path_to_event_file, duration_ms=50.0, start_index=0):
        print('Will use fixed duration event windows of size {:.2f} ms'.format(duration_ms))
        print('Output frame rate: {:.1f} Hz'.format(1000.0 / duration_ms))
        file_extension = splitext(path_to_event_file)[1]
        assert(file_extension in ['.txt', '.zip'])
        self.is_zip_file = (file_extension == '.zip')

        if self.is_zip_file:  # '.zip'
            self.zip_file = zipfile.ZipFile(path_to_event_file)
            files_in_archive = self.zip_file.namelist()
            assert(len(files_in_archive) == 1)  # make sure there is only one text file in the archive
            self.event_file = self.zip_file.open(files_in_archive[0], 'r')
        else:
            self.event_file = open(path_to_event_file, 'r')

        # ignore header + the first start_index lines
        for i in range(1 + start_index):
            self.event_file.readline()

        self.last_stamp = None
        self.duration_s = duration_ms / 1000.0

    def __iter__(self):
        return self

    def __del__(self):
        if self.is_zip_file:
            self.zip_file.close()

        self.event_file.close()

    def __next__(self):
        with Timer('Reading event window from file'):
            event_list = []
            for line in self.event_file:
                if self.is_zip_file:
                    line = line.decode("utf-8")
                t, x, y, pol = line.split(' ')
                t, x, y, pol = float(t), int(x), int(y), int(pol)
                event_list.append([t, x, y, pol])
                if self.last_stamp is None:
                    self.last_stamp = t
                if t > self.last_stamp + self.duration_s:
                    self.last_stamp = t
                    event_window = np.array(event_list)
                    return event_window

        raise StopIteration
