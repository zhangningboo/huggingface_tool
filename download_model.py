# -*- coding: utf-8 -*-
import os
import subprocess
from pathlib import Path
import argparse
import timeit


"""
Reference: https://hf-mirror.com
"""

os.environ['HF_ENDPOINT'] = rf"https://hf-mirror.com"
MODEL_SAVE_PATH = Path(os.path.realpath(__file__)).parent


def process_args():
    parser = argparse.ArgumentParser(
                    prog='download huggingface model',
                    description='Use hf-mirror.com as proxy',
                    epilog='Text at the bottom of help')
    parser.add_argument('-t', '--token', type=str, help='Set a token: https://huggingface.co/settings/tokens')
    parser.add_argument('-m', '--model', nargs='+', required=True, help='Model name, for example: microsoft/phi-2')
    parser.add_argument('-l', '--loop', type=int, default=42, help='Detault retry times.')
    return parser.parse_args()


def check_token(token: str):
    token_file = MODEL_SAVE_PATH.joinpath('.token')
    if token is None:
        if token_file.exists():
            with open(token_file, mode='r') as f:
                    token = f.readline()
    else:
        with open(token_file, mode='w') as f:
            f.writelines(token)
    return token


def download_model(cmd: list, retry_times=0) -> bool:
    success = False
    retry_times += 1
    for _ in range(retry_times):
        exec_res = subprocess.run(cmd)
        if exec_res.returncode == 0:
            success = True
            break
    return success


def save_download_res(content: str):
    res_file = MODEL_SAVE_PATH.joinpath('result.txt')
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(res_file, mode='a+') as f:
        f.writelines(f"[{timestamp}] {content}\n")


def download(opt: argparse.Namespace):
    try:
        subprocess.run(["pip", "install", "-U", "huggingface_hub"])
        retry_times = opt.loop
        for model_name in opt.model:
            org, model_version = model_name.split("/")
            model_dir = MODEL_SAVE_PATH.joinpath(org).joinpath(model_version)
            model_dir.mkdir(exist_ok=True, parents=True)
            token = check_token(opt.token)
            cmd = ["huggingface-cli", "download", "--resume-download", "--local-dir-use-symlinks", "False", model_name, "--local-dir", model_dir.absolute().as_posix()]
            if token is not None:
                cmd = [*cmd, "--token", token]
            success = download_model(cmd, retry_times=retry_times)
            save_download_res(f"{model_name}: {success}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    opt = process_args()
    print(f"{opt.token = }")
    print(f"{opt.model = }")
    download(opt)
