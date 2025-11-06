import os
import regex as re
from collections import defaultdict
from multiprocessing import Pool, Manager  # 导入多进程池
from functools import partial
# class Tokenizer :

from typing import BinaryIO

class BPETokenizer :
    def __init__(
        self, 
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
    ):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

    def train_bpe(
            self,
            **kwargs,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Dynamic update the bytes_pair_counts
        Process:
        1. vocabulary initialization
            A one-to-one mapping from bytestring token to integer ID
        2. pre-tokenization
            1) chunk boundaries
            2) remove special token
            3) run pre-tokenization
        3. compute BPE merges (consider how to optimize merge step)
            1) compute frequency of pairs
            2) merge pairs
            3) update
        """
        # 自定义比较类
        class PairItem:
            def __init__(self, count, pair):
                self.count = count
                self.pair = pair

            def __lt__(self, other):
                # 先按频率降序，频率相同时按pair降序
                return (self.count, self.pair) > (other.count, other.pair)# 频率相同时，pair大的优先

        # return self.train_bpe_slow()

        input_path = self.input_path
        vocab_size = self.vocab_size
        special_tokens = self.special_tokens
        # 1. init vocabulary
        vocabulary :dict[int, bytes] = {k : bytes([k]) for k in range(256)}
        cur_vocab_size = 256
        for special_token in special_tokens :
            vocabulary[cur_vocab_size] = special_token.encode('utf-8')
            cur_vocab_size += 1

        # 关键数据结构
        token_counts = defaultdict(int)
        pair_counts = defaultdict(int)
        bytes_to_tokens: dict[bytes, set[bytes]] = defaultdict(set)

        with open(input_path, "rb") as f:
            num_processes = os.cpu_count() or 4
            # 2. 1) get chunk boundaries
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            chunks = []
            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")

                # Run pre-tokenization on your chunk and store the counts for each pre-token
                chunks.append(chunk)
            # 2. 2) remove special tokens
            special_pat = re.compile("|".join(map(re.escape, special_tokens)))

            # 2. 3) run pre-tokenization
            partial_pre_token = partial(pre_tokenization,special_pat=special_pat)


            with Pool(processes=num_processes) as pool:
                results = pool.map(partial_pre_token, chunks)


            for result in results:
                for token, count in result.items():
                    # 这里不仅要建立正向链接，还要建立反向链接

                    token_bytes = token.encode('utf-8')
                    bytes_tuple = tuple(bytes([b]) for b in token_bytes)
                    for b1, b2 in zip(bytes_tuple[:-1], bytes_tuple[1:]) :
                        pair_counts[(b1, b2)] += count               # 统计每个 pair 的总数
                        bytes_to_tokens[(b1, b2)].add(bytes_tuple)   # 为每个 pair 建立反向链接，找到被影响的 token

                    token_counts[bytes_tuple] += count

        # 3. compute BPE merges
        merges_pair_list = []
        import heapq
        heap = [PairItem(count, pair) for pair, count in pair_counts.items() if count > 0]
        heapq.heapify(heap)

        # Track which pairs have been updated to avoid duplicate heap entries
        pairs_updated_this_iteration = set()

        while cur_vocab_size != vocab_size and heap:
            pair_item = heapq.heappop(heap)
            merge_pair = pair_item.pair
            merge_pair_count = pair_item.count

            # Skip stale entries (lazy deletion)
            if merge_pair_count != pair_counts[merge_pair] :
                continue
            if merge_pair_count == 0:
                break
            merges_pair_list.append(merge_pair)

            # Clear the set for this iteration
            pairs_updated_this_iteration.clear()

            # Update
            affected_tokens = bytes_to_tokens[merge_pair]

            for affected_token in list(affected_tokens) :
                if affected_token not in token_counts:
                    continue
                affected_token_count = token_counts[affected_token]

                # Remove old pairs (except the merge_pair itself, which we'll delete)
                for b1, b2 in zip(affected_token[:-1], affected_token[1:]):
                    if (b1, b2) != merge_pair:  # Don't update the pair we're merging
                        bytes_to_tokens[(b1, b2)].discard(affected_token)
                        pair_counts[(b1, b2)] -= affected_token_count
                        pairs_updated_this_iteration.add((b1, b2))

                # Create new token after merge
                new_token = get_new_token(affected_token, merge_pair)

                # Add new pairs
                for b1, b2 in zip(new_token[:-1], new_token[1:]):
                    bytes_to_tokens[(b1, b2)].add(new_token)
                    pair_counts[(b1, b2)] += affected_token_count
                    pairs_updated_this_iteration.add((b1, b2))

                # Update token counts
                del token_counts[affected_token]
                token_counts[new_token] += affected_token_count

            # Push updated pairs to heap ONCE per pair (not once per token!)
            for pair in pairs_updated_this_iteration:
                if pair_counts[pair] > 0:  # Only push if count is positive
                    heapq.heappush(heap, PairItem(pair_counts[pair], pair))

            # Clean up the merged pair
            del bytes_to_tokens[merge_pair]
            del pair_counts[merge_pair]

            # FIX: Store merged bytes, not the pair tuple
            vocabulary[cur_vocab_size] = merge_pair[0] + merge_pair[1]
            cur_vocab_size += 1

        return (vocabulary, merges_pair_list)

    def train_bpe_slow(
        self,
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
            1) compute frequency of pairs
            2) merge pairs
            3) update 
        """
        input_path = self.input_path
        vocab_size = self.vocab_size
        special_tokens = self.special_tokens

        # 1. init vocabulary
        vocabulary :dict[int, bytes] = {k : bytes([k]) for k in range(256)}
        cur_vocab_size = 256
        for special_token in special_tokens :
            vocabulary[cur_vocab_size] = special_token.encode('utf-8')
            cur_vocab_size += 1
        
        total_token_counts = defaultdict(int)
        with open(input_path, "rb") as f:
            num_processes = os.cpu_count() or 4
            # 2. 1) get chunk boundaries
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            chunks = []
            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                chunks.append(chunk)
            # 2. 2) remove special tokens
            special_pat = re.compile("|".join(map(re.escape, special_tokens)))

            # 2. 3) run pre-tokenization
            partial_pre_token = partial(pre_tokenization,special_pat=special_pat)

            
            with Pool(processes=num_processes) as pool:
                results = pool.map(partial_pre_token, chunks)

            
            for result in results:
                for token, count in result.items():
                    # 这里可能有问题：将字符逐个编码为字节
                    # bytes_tuple = tuple(char.encode('utf-8') for char in token)
                    # 直接将整个字符串编码为字节，然后转换为字节元组
                    token_bytes = token.encode('utf-8')
                    bytes_tuple = tuple(bytes([b]) for b in token_bytes)
                    total_token_counts[bytes_tuple] += count
            
        # 3. compute BPE merges
        merges_pair_list = []
        while cur_vocab_size < vocab_size :
            merges_pair = get_max_frequency_bytes_tuple(total_token_counts=total_token_counts)
            if merges_pair == (b'', b'') :
                break
            merges_pair_list.append(merges_pair)
            vocabulary[cur_vocab_size] = merges_pair[0] + merges_pair[1]
            total_token_counts = update_token_counts(total_token_counts, merges_pair)
            cur_vocab_size += 1


        return vocabulary, merges_pair_list     
        
        

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
        special_pat: re.Pattern,) -> dict[str, int] :
    """
    A coarse-grained tokenization. 
    Get a dict[str,str_cnt]
    1. remove special tokens
    2. pre-tokenization with the regex pattern
    """

    # remove special tokens
    sub_chunks = special_pat.split(chunk)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    counts = defaultdict(int)
    for sub_chunk in sub_chunks:
        matches = re.finditer(PAT, sub_chunk)
        for m in matches:
            # 关键修改：用 m.group() 获取匹配的字符串，替代不可序列化的 m
            token_str = m.group()
            counts[token_str] += 1

    return counts
    
def get_max_frequency_bytes_tuple(total_token_counts:dict[tuple[bytes], int]) -> tuple[bytes,bytes] :
    counts = defaultdict(int)
    for token, times in total_token_counts.items() :
        for b1, b2 in zip(token[:-1], token[1:]) :
            counts[(b1,b2)] += times
    if not counts:
        return (b'', b'')  # 或抛出异常
    
    max_pair = max(counts.items(), key=lambda x: (x[1], x[0]))[0]
    return max_pair

def get_new_token(old_token: tuple[bytes], merge_pair: tuple[bytes, bytes]) -> tuple[bytes] :
    b1, b2 = merge_pair
    merge_bytes = b1 + b2
    new_token = []
    i = 0
    while i < len(old_token) - 1:
        now_token_pair = old_token[i] + old_token[i + 1]
        if now_token_pair == merge_bytes:
            new_token.append(merge_bytes)
            i += 2
        else :
            new_token.append(old_token[i])
            i += 1
    # Don't forget the last token if we didn't merge it
    if i < len(old_token):
        new_token.append(old_token[i])

    return tuple(new_token)
def update_token_counts(total_token_counts: dict[tuple[bytes], int],
                        merge_pair: tuple[bytes, bytes]) -> dict[tuple[bytes], int] :
    """
    merge the bytes to update the counts dict
    """
    new_token_counts = defaultdict(int)
    b1, b2 = merge_pair
    merge_bytes = b1 + b2
    for token, count in total_token_counts.items() :
        if len(token) < 2 :
            continue

        new_token = []
        i = 0
        while i < len(token) - 1:
            now_token_pair = token[i] + token[i + 1]
            if now_token_pair == merge_bytes:
                new_token.append(merge_bytes)
                i += 2
            else :
                new_token.append(token[i])
                i += 1
        if i < len(token) :                     # 这里忘记把最后一个 bytes 加入
            new_token.append(token[i])
        new_token_tuple = tuple(new_token)
        new_token_counts[new_token_tuple] += count
    
    return dict(new_token_counts)



if __name__ == "__main__":
    tokenizer = BPETokenizer(
        input_path="/home/marklee/study/cs336/assignment1-basics/data/test.txt",
        vocab_size=260,
        special_tokens=["<|endoftext|>"]
    )
    vocab, merges = tokenizer.train_bpe()

    