import os
import re
import sys
import h5py
import gzip
import tqdm
import time
import pathlib
import argparse

import numpy as np
import pandas as pd

import torch

from Bio import SeqIO

from sklearn.utils import shuffle
from typing import Tuple, Dict, Union

from MCIENet.utils.helper_func import saveJson, loadJson, chrom_to_int

def get_args():
    parser = argparse.ArgumentParser(
        description="generate hdf5 files for prediction."
        + " The validation chromosomes are 5, 14."
        + " The test chromosomes are 4, 7, 8, 11. Rest will be training."
    )
    # input data
    parser.add_argument("--genome_file", required=True, help="The fasta file of reference genome.")
    parser.add_argument("--pos_files", nargs="*", default=[], help="The positive files")
    parser.add_argument("--neg_files", nargs="*", default=[], help="The negative files")

    # output
    # parser.add_argument( "-n", "--name", type=str, required=True, help="The prefix of the output files.")
    parser.add_argument("-o", "--out_dir", type=str, required=True, help="The output directory.")

    # dna encode args
    parser.add_argument("-s", "--size", type=int, required=True, help="encode size of anchors")
    parser.add_argument("--min_size", type=int, required=True, help="minimum size of anchors to use")

    parser.add_argument("-m","--method", type=str, required=False, help="onehot、dnabert、text")

    ## DNABERT arg 
    parser.add_argument(
        "--pool", type=str, default='mean', help="pool method in DNAbert"
    )
    parser.add_argument(
        "--warm_up", type=int, default=50, help="warm up for DNAbert (alibi size)"
    )
    return parser.parse_args()


class DNAseq:
    def __init__(self, fasta_path: str, ftype: str = None, encode_method: str = 'onehot'):
        super().__init__()
        self.dna_seq = self._load_fasta(fasta_path, ftype)
        self.length = [len(i) for i in self.dna_seq]
        self.encode_method = encode_method
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if encode_method == 'dnabert':
            print('load danbert model...')
            self._mount_danbert_model(model="zhihan1996/DNABERT-2-117M", model_path="pretrain/DNABERT-2-117M")

    def _load_fasta(self, path: str, ftype: str = None) -> list:
        start = time.time()
        if ftype == None:
            ftype = "." + path.split(".")[-1]
        else:
            assert path[-3:] == ftype, f"fasta file doesn't match file type{ftype}"

        if ftype == ".fa":
            _open = open(path, "rt")  # 換行方式: \r\n -> \n
        elif ftype == ".gz":
            _open = gzip.open(path, "rt", encoding="utf8")
        else:
            raise ValueError("file type is not recognized. (Support type: .fa, .gz)")

        dna_seq = ["" for i in range(24)]
        for o in SeqIO.parse(_open, "fasta"):
            if o.name.replace("chr", "") not in list(map(str, range(1, 23))) + ["X"]:
                continue
            if o.name == "chrX":
                temp_key = 23
            else:
                temp_key = int(o.name.replace("chr", ""))

            dna_seq[temp_key] = str(o.seq)

        _open.close()

        print(f"timeiuse: {round(time.time() - start)}\n")

        return dna_seq

    def _mount_danbert_model(self, model:str, model_path:str) -> None:
        from transformers import AutoTokenizer, AutoModel
        
        self.tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model, cache_dir=model_path, trust_remote_code=True, low_cpu_mem_usage=True).to(self.device)

    def encode_pairs(
        self,
        pair: list,
        chr1: int,
        chr2: int,
        s1: int,
        e1: int,
        s2: int,
        e2: int,
        size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        anchor1 = self.encode_seq(pair[chr1], pair[s1], pair[e1], size)
        anchor2 = self.encode_seq(pair[chr2], pair[s2], pair[e2], size)

        return anchor1, anchor2

    def encode_seq(self, chrom: int, start: int, end: int, size: int) -> np.ndarray:

        # Get fixed anchor range
        seq_len = self.length[chrom]

        if start < 0 or end > seq_len:
            return None

        center = start + (end - start) / 2

        start_ = center - size / 2
        end_ = center + size / 2

        if start_ < 0:
            start_ = 0
            end_ = size
        elif end_ > seq_len:
            start_ = seq_len - size
            end_ = seq_len

        # get anchor seq
        seq = self.dna_seq[chrom][int(start_) : int(end_)]
        if seq is None:
            return None

        # encode anchor seq
        if self.encode_method == 'onehot':
            mat = self.get_seq_encode(seq, "channels_first", one_d=True, rc=False) # 4*size
        elif self.encode_method == 'dnabert':
            mat = self.danbert_encode(seq) # 768
        else:
            mat = seq

        # parts = []
        # for i in range(0, len(seq), 500):
        #     if i + 1000 >= len(seq):
        #         break
        #     parts.append(mat[:, i:i + 1000])
        # parts.append(mat[:, -1000:])
        # parts = np.array(parts, dtype='float32')

        return mat

    def get_seq_encode(
        self, seq: str, data_format: str, one_d: bool, rc: bool = False
    ) -> np.ndarray:
        channels = 4
        seq_len = len(seq)
        mat = np.zeros((seq_len, channels), dtype="float32")

        for i, a in enumerate(seq):
            idx = i
            if idx >= seq_len:
                break
            a = a.lower()

            if a == "a":
                mat[idx, 0] = 1
            elif a == "g":
                mat[idx, 1] = 1
            elif a == "c":
                mat[idx, 2] = 1
            elif a == "t":
                mat[idx, 3] = 1
            else:
                mat[idx, 0:4] = 0

        if rc:
            mat = mat[::-1, ::-1]

        if not one_d:
            mat = mat.reshape((1, seq_len, channels))

        if data_format == "channels_first":
            axes_order = [
                len(mat.shape) - 1,
            ] + [i for i in range(len(mat.shape) - 1)]
            mat = mat.transpose(axes_order)

        return mat

    def danbert_encode(self, dna_seq:str, pool:str='mean') -> np.ndarray:
        inputs = self.tokenizer(dna_seq, return_tensors = 'pt')["input_ids"]
        inputs = inputs.to(self.device)
        hidden_states = self.model(inputs)[0] # [1, sequence_length, 768]

        if pool == 'mean':
            # embedding with mean pooling
            embedding = torch.mean(hidden_states[0], dim=0)
        elif pool == 'max':
            # embedding with max pooling
            embedding = torch.max(hidden_states[0], dim=0)[0]
        else:
            raise ValueError('pool moust be mean or max.')

        # print(embedding.shape) # expect to be 768
        
        return embedding.detach().cpu().numpy()

def loop_filter(
    row: str,
    chrom_len: list,
    label: int,
    min_size: int,
    int_cols: list = [1, 2, 4, 5],
    chrom_cols: list = [0, 3],
) -> Tuple[bool, Union[str, list]]:
    """
    Process a loop (one row in pos & neg sample file), including data category, anchor order,
    and whether the loop meets the standards
    """

    tokens = row.strip().split()  # \t 分割

    assert len(tokens) + 1 > len(
        int_cols + chrom_cols
    ), f"The input data format cannot be recognized: {tokens}"

    chr1, chr2 = chrom_cols  # There is an interaction between chr1 and chr2, chr1 may be the same as chr2.

    s1, e1, s2, e2 = int_cols  # s1 and e1 are the start and end positions of chr1 respectively.

    # chrom filtering and data type conversion
    for i in chrom_cols:
        if not re.match("^chr(\d+|X)$", tokens[i]):
            return False, f"wrong chrom format | {chr1}, {chr2}"

        tokens[i] = chrom_to_int(tokens[i])

        if not (0 < tokens[i] <= len(chrom_len)):
            return False, f"wrong chrom (chr1 ~ 23, chrX) | {chr1}, {chr2}"

    for i in int_cols:
        tokens[i] = int(tokens[i])

    if tokens[chr1] != tokens[chr2]:
        return False, "inter-chrom interaction"

    if min((tokens[e1] - tokens[s1]), (tokens[e1] - tokens[s1])) < min_size:
        return False, "Anchor lengh less than min_size"

    # change chromosome order
    if tokens[s1] > tokens[s2]:
        temp = tokens[chr2], tokens[s2], tokens[e2]
        tokens[chr2], tokens[s2], tokens[e2] = tokens[chr1], tokens[s1], tokens[e1]
        tokens[chr1], tokens[s1], tokens[e1] = temp

    if tokens[e1] >= chrom_len[tokens[chr1]] or tokens[e2] > chrom_len[tokens[chr2]]:
        return False, "region out of range"

    if tokens[s2] < tokens[e1]:
        return False, "anchor overlap"

    # distance range 5k ~ 2000k
    if (tokens[chr1] == tokens[chr2]) and not (
        5000 <= (tokens[s2] - tokens[s1] + tokens[e2] - tokens[e1]) / 2 <= 2000000
    ):
        return False, "anchor distance not in range (5k ~ 2000k)"

    if len(tokens) < 7:
        tokens.append(label)
    else:
        tokens[6] = int(float(tokens[6]))

    return True, tokens


def load_pairs(
    pos_files: list, neg_files: list, chrom_len: list, out_dir: str, min_size: int
) -> Dict[str, Dict[str, list]]:
    start = time.time()
    train_pairs = []
    train_labels = []
    val_pairs = []
    val_labels = []
    test_pairs = []
    test_labels = []

    error_ls = []
    info_dt = {}

    val_chroms = [5, 14]
    test_chroms = [4, 11, 7, 8]

    files = pos_files + neg_files

    for fn in files:
        label = 1 if fn in pos_files else 0

        with open(fn) as f:
            for r in f:
                result, tokens = loop_filter(r, chrom_len, label, min_size)
                if result:
                    if tokens[0] in val_chroms:
                        val_pairs.append(tokens)
                        val_labels.append(label)
                    elif tokens[0] in test_chroms:
                        test_pairs.append(tokens)
                        test_labels.append(label)
                    else:
                        train_pairs.append(tokens)
                        train_labels.append(label)
                else:
                    error_ls.append([fn, r, tokens])

    error_df = pd.DataFrame(error_ls, columns=["file", "row", "filter type"])
    error_df["distance"] = (
        error_df["row"]
        .str.split()
        .apply(lambda x: (int(x[4]) + int(x[5]) - int(x[1]) + int(x[2])) / 2)
    )
    error_df["row"] = error_df["row"].str.replace("\t", " | ")
    error_df.to_csv(os.path.join(out_dir, "error_pairs.csv"), index=False)

    info_dt["error_info"] = {}
    info_dt["error_info"]["file_count"] = error_df["file"].value_counts().to_dict()
    info_dt["error_info"]["filter_type"] = (
        error_df["filter type"].value_counts().to_dict()
    )

    train_pairs, train_labels = shuffle(train_pairs, train_labels)
    val_pairs, val_labels = shuffle(val_pairs, val_labels)
    test_pairs, test_labels = shuffle(test_pairs, test_labels)

    pair_dt = {
        "train": {"pairs": train_pairs, "labels": train_labels},
        "val": {"pairs": val_pairs, "labels": val_labels},
        "test": {"pairs": test_pairs, "labels": test_labels},
    }

    # saveJson(pair_dt, os.path.join(out_dir, "pair_dt.json"))

    info_dt["data_num"] = {k: len(v["labels"]) for k, v in pair_dt.items()}

    saveJson(info_dt, os.path.join(out_dir, "info_dt.json"))

    print(f"timeiuse: {round(time.time() - start)}\n")

    return pair_dt


def creat_out_hander(path: str, group_ls: list, overlap: bool=False) -> Tuple[h5py.File, dict]:
    if overlap:
        pathlib.Path(path).unlink(missing_ok=True)

    if os.path.exists(path):
        f = h5py.File(path, "a")
    else:
        f = h5py.File(path, "w")

    record_dt = {}
    for name in group_ls:
        if name not in f:
            f.create_group(name)
        if 'data' in f[name]:
            record_dt[name] = f[name]['data'].shape[0]
        else:
            record_dt[name] = 0

    return f, record_dt

# ==============================================


def main():
    args = get_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if len(args.pos_files) <= 0 and len(args.neg_files) <= 0:
        print("Nothing to do")
        sys.exit(0)

    print("Loading DNA-seq from fasta file...")
    dna_seq = DNAseq(args.genome_file, encode_method=args.method)

    print("Loading pairs...")
    pair_dt = load_pairs(
        args.pos_files, args.neg_files, dna_seq.length, args.out_dir, args.min_size
    )

    out_hander, data_proc_dt = creat_out_hander(os.path.join(args.out_dir, "data.h5"), pair_dt.keys())

    if args.method == 'dnabert':
        print(f'warming up dnabert')
        for pairs in tqdm.tqdm(pair_dt['train']["pairs"][:args.warm_up]):
            _ = dna_seq.encode_pairs(
                pairs, chr1=0, chr2=3, s1=1, e1=2, s2=4, e2=5, size=args.size
            )
    
    skip_count = 0

    for name in data_proc_dt.keys():
        print(f"encoding and output {name} data")
        if "labels" not in out_hander[name]:
            out_hander[name].create_dataset(
                "labels",
                data=pair_dt[name]["labels"],
                dtype="uint8",
                chunks=True,
                compression="gzip"
            )

        num_done = data_proc_dt[name]
        total = len(pair_dt[name]["pairs"])

        pair_df = pd.DataFrame(pair_dt[name]["pairs"], columns=['chr1', 's1', 'e1', 'chr2', 's2', 'e2', 'interaction'])
        pair_df.to_csv(os.path.join(args.out_dir, f'{name}_pairs.csv'), index=False)

        if total == num_done:
            print(f'{name} data proccess already done, skip! ({num_done} pairs in total)\n')
            continue
        else:
            print(f'{num_done} {name} data have been processed, leaving {total - num_done} pairs to be processed\n')

        pairs_ls = [pairs for idx, pairs in enumerate(pair_dt[name]["pairs"]) if idx >= num_done]

        print(f'encoding and output {name} data')
        for pairs in tqdm.tqdm(pairs_ls):
            anchor1, anchor2 = dna_seq.encode_pairs(
                pairs, chr1=0, chr2=3, s1=1, e1=2, s2=4, e2=5, size=args.size
            )

            if (anchor1 is None or anchor2 is None):
                print(f'skip - {pairs}')
                skip_count +=1
                continue

            seq = np.expand_dims(np.array((anchor1, anchor2)), 0) 
            # onehot: (1, 2, 4, size) | dnabert: (1, 2, 768) | text:  (1, 2, size)

            if "data" not in out_hander[name]:
                if args.method == 'onehot':
                    out_hander[name].create_dataset(
                        "data",
                        data=seq,
                        dtype="uint8",
                        chunks=True,
                        compression="gzip",
                        maxshape=(None, 2, 4, args.size),
                    )
                elif args.method == 'dnabert':
                    out_hander[name].create_dataset(
                        "data",
                        data=seq,
                        dtype="float32",
                        chunks=True,
                        compression="gzip",
                        maxshape=(None, 2, 768),
                    )
                elif args.method == 'text':
                    out_hander[name].create_dataset(
                        "data",
                        data=seq.astype('S'),
                        dtype=h5py.string_dtype(encoding='utf-8'),
                        chunks=True,
                        compression="gzip",
                        maxshape=(None, 2),
                    )
                    
                else:
                    raise ValueError('Unsupported method')
            else:
                tar = out_hander[name]["data"]
                tar.resize((tar.shape[0] + seq.shape[0]), axis=0)
                tar[-seq.shape[0] :] = seq

    # "uint8": Unsigned integer (0 to 255)
    # "u8": 64-bit unsigned integer
    print(f'skip count: {skip_count}')
    out_hander.close()


if __name__ == "__main__":
    main()
