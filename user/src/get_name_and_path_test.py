from typing import Union, Tuple, List
import os
import re


def test_re():
    path_like = r"/home/tank/dxa/test/test.pb"
    pattern = r"[/]"
    ret = re.finditer(pattern, path_like)
    split_position = list(ret)[-1].span()[0]
    path = path_like[:split_position]
    file = path_like[split_position+1:]
    return path, file


def spilt_path_file(path_like: str) -> Tuple[str, str]:
    pattern = r"[/]"
    ret = re.finditer(pattern, path_like)
    split_position = list(ret)[-1].span()[0]
    path = path_like[:split_position]
    file = path_like[split_position + 1:]
    return path, file


if __name__ == '__main__':
    print(test_re())
