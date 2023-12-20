# -*- coding: utf-8 -*-
import os
import subprocess
from pathlib import Path
import argparse


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


def download(opt: argparse.Namespace):
    try:
        subprocess.run(["pip", "install", "-U", "huggingface_hub"])
        for model_name in opt.model:
            org, model_version = model_name.split("/")
            model_dir = MODEL_SAVE_PATH.joinpath(org).joinpath(model_version)
            model_dir.mkdir(exist_ok=True, parents=True)
            token = check_token(opt.token)
            if token is None:
                subprocess.run(["huggingface-cli", "download", "--resume-download", "--local-dir-use-symlinks", "False", model_name, "--local-dir", model_dir.absolute().as_posix()])
            else:
                subprocess.run(["huggingface-cli", "download", "--token", token, "--resume-download", "--local-dir-use-symlinks", "False", model_name, "--local-dir", model_dir.absolute().as_posix()])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    opt = process_args()
    print(f"{opt.token = }")
    print(f"{opt.model = }")
    download(opt)


# huggingface-cli download --token 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 --resume-download --local-dir-use-symlinks False Qwen/Qwen-72B-Chat --local-dir Qwen-72B-Chat
