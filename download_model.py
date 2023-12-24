# -*- coding: utf-8 -*-
import os
import subprocess
from pathlib import Path
import argparse
import timeit
import time


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
    parser.add_argument('-p', '--project', nargs='+', required=True, help='Project name, for example: microsoft/phi-2 Stevross/mmlu')
    parser.add_argument('-l', '--loop', type=int, default=42, help='Detault retry times.')
    parser.add_argument('--type', type=str, default='model', choices=['model', 'dataset', 'space'], help='Detault retry times.')
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


def download_project(cmd: list, retry_times=0) -> bool:
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
        project_type = opt.type
        dst_path = MODEL_SAVE_PATH.joinpath(project_type)
        for project_name in opt.project:
            org, version = project_name.split("/")
            project_path = dst_path.joinpath(org).joinpath(version)
            project_path.mkdir(exist_ok=True, parents=True)
            token = check_token(opt.token)
            cmd = ["huggingface-cli", "download", "--repo-type", project_type, "--resume-download", "--local-dir-use-symlinks", "False", project_name, "--local-dir", project_path.absolute().as_posix()]
            if token is not None:
                cmd = [*cmd, "--token", token]
            success = download_project(cmd, retry_times=retry_times)
            save_download_res(f"[{project_type.upper()}] {project_name}: {success}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    opt = process_args()
    print(f"{opt.token = }")
    print(f"{opt.project = }")
    download(opt)
