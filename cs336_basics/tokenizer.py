import os
import regex as re
from collections import defaultdict
# class Tokenizer :

from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenization(
        chunk: str, 
        special_pat: re.Pattern) -> dict[str,int] :
    """
    A coarse-grained tokenization. 
    Get a dict[str,str_cnt]
    1. remove special tokens
    2. pre-tokenization with the regex pattern
    """

    # remove special tokens
    sub_chunks = special_pat.split(chunk)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    ret: defaultdict[str, int]= defaultdict(int)
    for sub_chunk in sub_chunks:
        matches = re.finditer(PAT, sub_chunk)
        for m in matches:
            ret[m] += 1
    
    return ret


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Process:
    1. vocabulary initialization
        A one-to-one mapping from bytestring token to integer ID
    2. pre-tokenization
        1) chunk boundaries
        2) remove special token
        3) run pre-tokenization
    3. compute BPE merges (consider how to optimize merge step)
    """
    # 1. init vocabulary
    vocabulary :dict[int, bytes] = {k : bytes([k]) for k in range(256)}
    cur_vocab_size = 256
    for special_token in special_tokens :
        vocabulary[cur_vocab_size] = special_token.encode('utf-8')
    
    # 2. 1) get chunk boundaries
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            special_pat = re.compile("|".join(map(re.escape, special_tokens)))
            # Run pre-tokenization on your chunk and store the counts for each pre-token



    pass

if __name__ == "__main__":
    s : str = "hello"
    for b in s.encode('utf-8') :
        print(b)