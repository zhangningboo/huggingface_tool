### Download model

``` shell
$ python download_model.py -t hf_xxxxxxx -p Microsoft/phi-2 Microsoft/phi-1  # If you have a token
# or
$ python download_model.py -p Microsoft/phi-2 Microsoft/phi-1
```

### Download dataset

``` shell
$ python download_model.py -t hf_xxxxxxx -p Stevross/mmlu  # If you have a token
# or
$ python download_model.py --type dataset -p Stevross/mmlu
```

#### ⚠️Notice: The token you provided will be saved to the file .token in the script folder!
