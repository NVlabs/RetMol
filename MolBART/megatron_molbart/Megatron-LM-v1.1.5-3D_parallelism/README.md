This is a snapshot of Megatron v1.1.5 integrated with DeepSpeed's pipeline- and data-parallel training. This 3D parallelism integration
can train a model with [one trillion parameters](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/) using as few as 800 NVIDIA V100 GPUs.

See `examples/ds_pretrain_gpt2_pipe.sh` for an entry point to training with 3D parallelism.

See our [pull request](https://github.com/jeffra/DSE/pull/10) for a more detailed view of the integration.


[Megatron](https://arxiv.org/pdf/1909.08053.pdf) is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA. This repository is for ongoing research on training large transformer language models at scale. We developed efficient, model-parallel, and multinode training of [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and [BERT](https://arxiv.org/pdf/1810.04805.pdf) using mixed precision.

Using our GPT-2 model we achieve a perplexity of 10.8 on the WikiText-103 dataset (improving SOTA from 15.8) and an accuracy of 66.5% on the LAMBADA datasets. For BERT training, we swapped the position of the layer normalization and the residual connection in the model architecture (similar to GPT-2 architucture), which allowed the models to continue to improve as they were scaled up. Our BERT models with 3.9 billion parameters reaches a loss of 1.16, SQuAD 2.0 F1-score of 91.7, and RACE accuracy of 90.9%.

Our codebase is capable of efficiently training very large (several billion parameter) language models with both model and data parallelism. To demonstrate how the code scales with multiple GPUs we consider the following GPT-2 model sizes. All models use a vocabulary size of 51,200 and a sequence length of 1024.

![Cases](images/cases.png)

The table below details the weak scaling from 1 to 8 GPUs of our model parallelism code in both a DGX-2 and a DGX-A100. Notice that we double the batch size on the DGX-A100 but the iteration time decreases compared to the DGX-2 resulting in a **2.1x** speedup for the end-to-end application.

![Model Parallel Scaling](images/scaling-mp.png)

The following table details how Megatron scales using data parallelism in conjuction with model parallelism in a cluster of DGX-A100s. All of these cases use 128-way data parallelism and the scaling numbers are relative to a single A100 (Case 1B with a 1076ms iteration time).

![Data Parallel Scaling](images/scaling-dp.png)

<a id="contents"></a>
# Contents
<!-- MarkdownTOC -->

- [Setup](#setup)
  - [Downloading Checkpoints](#downloading-checkpoints)
- [Usage](#usage)
- [Training](#training)
  - [Data Preprocessing](#data-preprocessing)
  - [BERT Pretraining](#bert-pretraining)
  - [GPT-2 Pretraining](#gpt-2-pretraining)
  - [Distributed BERT or GPT-2 Pretraining](#distributed-bert-or-gpt-2-pretraining)
- [REALM Pipeline](#realm)
- [Evaluation and Tasks](#evaluation-and-tasks)
  - [GPT-2 Text Generation](#gpt-2-text-generation)
  - [GPT-2 Evaluation](#gpt-2-evaluation)
    - [WikiText Perplexity Evaluation](#wikitext-perplexity-evaluation)
    - [LAMBADA Cloze Accuracy](#lambada-cloze-accuracy)
  - [BERT Task Evaluation](#bert-task-evaluation)
    - [RACE Evaluation](#race-evaluation)
    - [MNLI Evaluation](#mnli-evaluation)
- [Datasets](#datasets)
  - [Collecting Wikipedia Training Data](#collecting-wikipedia-training-data)
  - [Collecting GPT-2 Webtext Data](#collecting-gpt-2-webtext-data)

<!-- /MarkdownTOC -->

<a id="setup"></a>
# Setup
We officially support only python 3.6, pytorch 1.5, cuda 10, and nccl 2.6 versions and above.

To use this repo please install the latest supported versions of PyTorch with GPU support and NVIDIA [APEX](https://github.com/NVIDIA/apex#quick-start). We strongly recommend using one of [NGC's recent PyTorch containers](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) (the latest compatible version at time of publication can be pulled with `docker pull nvcr.io/nvidia/pytorch:20.03-py3`). Data preprocessing requires [NLTK](https://www.nltk.org/install.html), though this is not required for training, evaluation or downstream tasks.

To use megatron you can either clone the repo or install it via pip (make sure python3-dev is installed):
<pre>
pip install megatron-lm
</pre>

<a id="downloading-checkpoints"></a>
## Downloading Checkpoints
We've provided two pretrained checkpoints for use to evaluate or finetuning downstream tasks. To access these checkpoints, first please [sign up](https://ngc.nvidia.com/signup) for and [setup](https://ngc.nvidia.com/setup/installers/cli) the NVIDIA GPU Cloud (NGC) Registry CLI.

The checkpoints can be downloaded with:
<pre>
ngc registry model download-version --dest &#60;output_base_directory&#62; nvidia/&#60;model_name&#62;:&#60;version&#62;
</pre>

The available models along with `<model_name>:<version>` are below:
* [BERT-345M](https://ngc.nvidia.com/catalog/models/nvidia:megatron_bert_345m): megatron\_bert\_345m:v0.0
* [GPT-2-345M](https://ngc.nvidia.com/catalog/models/nvidia:megatron_lm_345m): megatron\_lm\_345m:v0.0

The models require vocabulary files to run. The BERT uncased WordPiece vocab file can be extracted from Google's [pretrained BERT models](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt). The GPT-2 [vocab file](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json) and [merge table](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt) can be downloaded directly.

Further documentation for downloading models can be found in the [NGC documentation](https://docs.nvidia.com/dgx/ngc-registry-cli-user-guide/index.html#topic_6_4_1)

<a id="usage"></a>
# Usage

After installation, there are several possible workflows. The most comprehensive is:
1. Data preprocessing
2. Pretraining
3. Finetuning (Optional for zero-shot tasks)
4. Downstream task evaluation or text generation

However, steps 1 and 2 can be replaced by using one of the pretrained models mentioned above.

We've provided several scripts for pretraining both BERT and GPT-2 in [`examples`](./examples) directory, as well as scripts for both zero-shot and fine-tuned downstream tasks including MNLI, RACE, WikiText103, and LAMBADA evaluation. There is also a script for GPT-2 interactive text generation.

<a id="training"></a>
# Training
<a id="data-preprocessing"></a>
## Data Preprocessing
We support three file formats for training, but all require preprocessing. First, place your training data in a loose json format, with one json containing a text sample per line. For example:
<pre>
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
</pre>

The name of the `text` field of the json can be changed by using the `--json-key` flag in [`preprocess_data.py`](./tools/preprocess_data.py) The other metadata are optional and are not used in training.

The loose json is then processed into a binary format for training. To convert the json into mmap, cached index file, or the lazy loader format use `preprocess_data.py`. Set the `--dataset-impl` flag to `mmap`, `cached`, or `lazy`, respectively (default is `mmap`). An example script to prepare data for BERT training is:
<pre>
python tools/preprocess_data.py \
       --input my-corpus.json \
       --output-prefix my-bert \
       --vocab bert-vocab.txt \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences
</pre>

The output will be two files named, in this case, `my-bert_text_sentence.bin` and `my-bert_text_sentence.idx`. The `--data-path` specified in later BERT training is the full path and new filename, but without the file extension.

Some minor modifications are required for GPT-2 data preprocessing, namely, the addition of a merge table, an end-of-document token, removal of sentence splitting, and a change to the tokenizer type:
<pre>
python tools/preprocess_data.py \
       --input my-corpus.json \
       --output-prefix my-gpt2 \
       --vocab gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --append-eod
</pre>

Here the output files are named `my-gpt2_text_document.bin` and `my-gpt2_text_document.idx`. As before, in GPT-2 training, use the longer name without the extension as `--data-path`.

Further command line arguments are described in the source file [`preprocess_data.py`](./tools/preprocess_data.py).

<a id="bert-pretraining"></a>
## BERT Pretraining
`bash examples/pretrain_bert.sh`

This script runs single GPU 345M parameter BERT pretraining. Debugging is the primary use for single GPU training, as the code base and command line arguments are optimized for highly distributed training. Most of the arguments are fairly self-explanatory. By default, the learning rate decays linearly over the training iterations starting at `--lr` to a minimum set by `--min-lr` over `--lr-decay-iters` iterations. The fraction of training iterations used for warmup is set by `--warmup`. While this is single GPU training, the batch size specified by `--batch-size` is per GPU used for data parallelism. The data is partitioned into a 949:50:1 ratio for training/validation/test sets (default is 969:30:1). This partitioning happens on the fly, but is consistent across runs with the same random seed (1234 by default, or specified manually with `--seed`).

The logging, checkpoint-saving, and evaluation intervals are specified. Checkpointing the activations facilitates the training of larger models and/or batches. Note that the `--data-path` now includes the additional `_text_sentence` suffix added in preprocessing, but does not include the file extensions.

<pre>
CHECKPOINT_PATH=checkpoints/bert_345m
VOCAB_FILE=bert-vocab.txt
DATA_PATH=my-bert_text_sentence

BERT_ARGS="--num-layers 24 \
           --hidden-size 1024 \
           --num-attention-heads 16 \
           --seq-length 512 \
           --max-position-embeddings 512 \
           --lr 0.0001 \
           --train-iters 2000000 \
           --min-lr 0.00001 \
           --lr-decay-iters 990000 \
           --warmup 0.01 \
           --batch-size 8 \
           --vocab-file $VOCAB_FILE \
           --split 949,50,1 \
           --fp16"

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 100 \
             --eval-iters 10 \
             --checkpoint-activations"

python pretrain_bert.py \
       $BERT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH
</pre>

Further command line arguments are described in the source file [`arguments.py`](./megatron/arguments.py).

<a id="gpt-2-pretraining"></a>
## GPT-2 Pretraining
`bash examples/pretrain_gpt2.sh`

This script runs single GPU 345M parameter GPT-2 pretraining. As mentioned above, single GPU training is primarily intended for debugging purposes, as the code is optimized for distributed training.

It follows largely the same format as the previous BERT script with a few notable differences: the tokenization scheme used is BPE (which requires a merge table and a `json` vocabulary file) instead of WordPiece, the model architecture allows for longer sequences (note that the max position embedding must be greater than or equal to the maximum sequence length), and the `--lr-decay-style` has been set to cosine decay.  Note that the `--data-path` now includes the additional `_text_document` suffix added in preprocessing, but does not include the file extensions.

<pre>
CHECKPOINT_PATH=checkpoints/gpt2_345m
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=my-gpt2_text_document

GPT2_ARGS="--num-layers 24 \
           --hidden-size 1024 \
           --num-attention-heads 16 \
           --seq-length 1024 \
           --max-position-embeddings 1024 \
           --batch-size 4 \
           --lr 0.00015 \
           --train-iters 500000 \
           --lr-decay-iters 320000 \
           --lr-decay-style cosine \
           --vocab-file $VOCAB_FILE \
           --merge-file $MERGE_FILE \
           --warmup .01 \
           --fp16"

OUTPUT_ARGS=&#60;same as those in <a href="#bert-pretraining">BERT pretraining</a> above&#62;

python pretrain_gpt2.py \
       $GPT2_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
</pre>

Further command line arguments are described in the source file [`arguments.py`](./megatron/arguments.py).

<a id="distributed-bert-or-gpt-2-pretraining"></a>
## Distributed BERT or GPT-2 Pretraining
`bash examples/pretrain_bert_distributed.sh`

`bash examples/pretrain_gpt2_distributed.sh`

These scripts use the PyTorch distributed launcher for distributed training. As such, multinode training can be achieved by properly setting environment variables and using `init_method='env://'` in the launcher. See the official PyTorch [documentation](https://pytorch.org/docs/stable/distributed.html#launch-utility) for further description of these [environment variables](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization). By default, multinode training uses the [nccl](https://developer.nvidia.com/nccl) distributed backend. A simple set of additional arguments and the use of the PyTorch distributed module with the Python flag `-m torch.distributed.launch`, detailed below, are the only additional requirements to adopt distributed training.

The two tiers of parallelism are data and model parallelism. First, we facilitate two distributed data parallel implementations: a simple one of our own that performs gradient all-reduce at the end of back propagation step, and Torch's distributed data parallel wrapper that overlaps gradient reduction with back propagation computation. To switch between these two options use `--DDP-impl local` or `--DDP-impl torch`, respectively. As expected, Torch distributed data parallelism is more efficient at larger model parallel sizes. For example, for the 8.3 billion parameters model running on 512 GPUs, the scaling increases from 60% to 76% when Torch's distributed data parallel is used. However, the overlapping method requires more memory and for some configurations (e.g., 2.5 billion parameters using 2-way model parallel and 1.2 billion parameters with no model parallel) can make the overall training slower as a result. We empirically found that using a smaller model in those cases improves the training time.

Second, we developed a simple and efficient intra-layer model parallel approach. To use model parallelism, add the `--model-parallel-size` flag to specify the number of GPUs among which to split the model, along with the arguments passed to the distributed launcher as mentioned above. With `WORLD_SIZE` GPUs and `MP_SIZE` model parallel size, `WORLD_SIZE`/`MP_SIZE` GPUs will be used for data parallelism. The default value for `--model-parallel-size` is 1, which will not implement model parallelism.

Other than these minor changes, the distributed training is identical to the training on a single GPU.

Distributed BERT training:
<pre>
WORLD_SIZE=8
MP_SIZE=2

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT_PATH=checkpoints/bert_345m
VOCAB_FILE=bert-vocab.txt
DATA_PATH=my-bert_text_sentence
BERT_ARGS=&#60;same as those in <a href="#bert-pretraining">BERT pretraining</a> above&#62;
OUTPUT_ARGS=&#60;same as those in <a href="#bert-pretraining">BERT pretraining</a> above&#62;

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./pretrain_bert.py \
                $BERT_ARGS \
                $OUTPUT_ARGS \
                --save $CHECKPOINT_PATH \
                --load $CHECKPOINT_PATH \
                --data-path $DATA_PATH \
                --model-parallel-size $MP_SIZE \
                --DDP-impl torch
</pre>

Distributed GPT-2 training:
<pre>
WORLD_SIZE=8
MP_SIZE=2

DISTRIBUTED_ARGS=&#60;same as those directly above&#62;

CHECKPOINT_PATH=checkpoints/gpt2_345m
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=my-gpt2_text_document
GPT2_ARGS=&#60;same as those in <a href="#gpt-2-pretraining">GPT-2 pretraining</a> above&#62;
OUTPUT_ARGS=&#60;same as those in <a href="#bert-pretraining">BERT pretraining</a> above&#62;

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./pretrain_gpt2.py \
                $GPT2_ARGS \
                $OUTPUT_ARGS \
                --save $CHECKPOINT_PATH \
                --load $CHECKPOINT_PATH \
                --data-path $DATA_PATH \
                --model-parallel-size $MP_SIZE \
                --DDP-impl torch

</pre>

<a id="realm"></a>
## REALM Pipeline
We are working on implementing the [REALM](https://arxiv.org/pdf/2002.08909.pdf) system. The following sections (will) reflect the three stages of training it. For now it's just the ICT code.
Loosely, they are pretraining the retriever modules, then jointly training the language model and the retriever, and then finetuning a question answering head on the language model with fixed retriever.

### Inverse Cloze Task (ICT) Pretraining
1. Have a corpus in loose JSON format with the intention of creating a collection of fixed-size blocks of text as the fundamental units of data. For a corpus like Wikipedia, this will mean multiple sentences per block but also multiple blocks per document.
Run `tools/preprocess_data.py` to construct one or more indexed datasets with the `--split-sentences` argument to make sentences the basic unit. For the original REALM system, we construct two datasets, one with the title of every document, and another with the body.
Refer to the following script
<pre>
python preprocess_data.py \
    --input /path/to/corpus.json \
    --json-keys text title \
    --split-sentences \
    --tokenizer-type BertWordPieceLowerCase \
    --vocab-file /path/to/vocab.txt \
    --output-prefix corpus_indexed \
    --workers 5  # works well for 10 CPU cores. Scale up accordingly.
</pre>

2. Use a custom samples mapping function in place of `megatron/data/realm_dataset_utils.get_block_samples_mapping` if required. To do this, you will need to implement a new function in C++ inside of `megatron/data/helpers.cpp`. The samples mapping data structure is used to select the data that will constitute every training sample in advance of the training loop.
 The samples mapping is responsible for holding all of the required metadata needed to construct the sample from one or more indexed datasets. In REALM, the samples mapping contains the start and end sentence indices, as well as the document index (to find the correct title for a body) and a unique ID for every block.
3. Pretrain a BERT language model using `pretrain_bert.py`, with the sequence length equal to the block size in token ids. This model should be trained on the same indexed dataset that is used to supply the blocks for the information retrieval task.
In REALM, this is an uncased bert base model trained with the standard hyperparameters.
4. Use `pretrain_ict.py` to train an `ICTBertModel` which uses two BERT-based encoders to encode queries and blocks to perform retrieval with.
The script below trains the ICT model from REALM. It refrences a pretrained BERT model (step 3) in the `--bert-load` argument. The batch size used in the paper is 4096, so this would need to be run with data parallel world size 32.
<pre>
python pretrain_ict.py \
    --num-layers 12 \
    --num-attention-heads 12 \
    --hidden-size 768 \
    --batch-size 128 \
    --seq-length 256 \
    --max-position-embeddings 256 \
    --ict-head-size 128 \
    --train-iters 100000 \
    --checkpoint-activations \
    --bert-load /path/to/pretrained_bert \
    --load checkpoints \
    --save checkpoints \
    --data-path /path/to/indexed_dataset \
    --titles-data-path /path/to/titles_indexed_dataset \
    --vocab-file /path/to/vocab.txt \
    --lr 0.0001 \
    --num-workers 2 \
    --lr-decay-style linear \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --warmup .01 \
    --save-interval 3000 \
    --query-in-block-prob 0.1 \
    --fp16

</pre>

### Building an Index of Block Embeddings
After having trained an ICT model, you can now embed an entire dataset of blocks by creating a `BlockData` structure. After that has been saved, you can load it
and wrap it with a `FaissMIPSIndex` to do fast similarity search which is key in the learned information retrieval pipeline. The initial index can be built with the following script, meant to be run in an interactive session. It can leverage multiple GPUs on multiple nodes to index large datasets much more quickly.

<pre>
python tools/create_doc_index.py \
    --num-layers 12 \
    --hidden-size 768 \
    --ict-head-size 128 \
    --num-attention-heads 12 \
    --batch-size 128 \
    --checkpoint-activations \
    --seq-length 256 \
    --max-position-embeddings 256 \
    --ict-load /path/to/pretrained_ict \
    --data-path /path/to/indexed_dataset \
    --titles-data-path /path/to/titles_indexed_dataset \
    --block-data-path embedded_blocks.pkl \
    --indexer-log-interval 1000 \
    --indexer-batch-size 128 \
    --vocab-file /path/to/vocab.txt \
    --num-workers 2 \
    --fp16
</pre>

<a id="evaluation-and-tasks"></a>
# Evaluation and Tasks

We provide several command line arguments, detailed in the scripts listed below, to handle various zero-shot and fine-tuned downstream tasks. However, you can also finetune your model from a pretrained checkpoint on other corpora as desired. To do so, simply add the `--finetune` flag and adjust the input files and training parameters within the original training script. The iteration count will be reset to zero, and the optimizer and internal state will be reinitialized. If the fine-tuning is interrupted for any reason, be sure to remove the `--finetune` flag before continuing, otherwise the training will start again from the beginning.

Because evaluation requires substantially less memory than training, it may be advantageous to merge a model trained in parallel for use on a single GPU in downstream tasks. The following script accomplishes this.

<pre>
MODEL_PARALLEL_SIZE=2

VOCAB_FILE=bert-vocab.txt
CHECKPOINT_PATH=checkpoints/bert_345m

WORLD_SIZE=$MODEL_PARALLEL_SIZE python tools/merge_mp_partitions.py \
        --model-type BERT \
        --model-parallel-size $MODEL_PARALLEL_SIZE \
        --tokenizer-type BertWordPieceLowerCase \
        --vocab-file $VOCAB_FILE \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 512 \
        --max-position-embeddings 512 \
        --load $CHECKPOINT_PATH

</pre>

Several downstream tasks are described for both GPT-2 and BERT models below. They can be run in distributed and model parallel modes with the same changes used in the training scripts.

<a id="gpt-2-text-generation"></a>
## GPT-2 Text Generation
`bash examples/generate_text.sh`

We generate text samples using largely the GPT-2 pretraining script. Few changes need to make, such as we need to provide the path to the pretrained checkpoint, the length of the output samples, whether to generate texts unconditionally (`--num-samples` to denote how many samples to generate) or conditional (need to pass `--sample-input-file <filename>` where each line of the file will be used as the conditional texts). There are few optional parameters to play, e.g. `top-k`, `top-p`, or `greedy` (set top-k and top-p to 0) sampling..

<pre>
CHECKPOINT_PATH=checkpoints/gpt2_345m
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
GPT2_ARGS=&#60;same as those in <a href="#gpt-2-pretraining">GPT-2 pretraining</a> above&#62;

MAX_OUTPUT_SEQUENCE_LENGTH=1024
TEMPERATURE=1.0
TOP_P=0.9
NUMBER_OF_SAMPLES=2
OUTPUT_FILE=samples.json

python tools/generate_samples_gpt2.py \
       $GPT2_ARGS \
       --load $CHECKPOINT_PATH \
       --out-seq-length $MAX_OUTPUT_SEQUENCE_LENGTH \
       --temperature $TEMPERATURE \
       --genfile $OUTPUT_FILE \
       --num-samples $NUMBER_OF_SAMPLES \
       --top_p $TOP_P \
       --recompute
</pre>

<a id="gpt-2-evaluation"></a>
## GPT-2 Evaluation
We include example scripts for GPT-2 evaluation on WikiText perplexity evaluation and LAMBADA Cloze accuracy.

<a id="wikitext-perplexity-evaluation"></a>
### WikiText Perplexity Evaluation
For even comparison with prior works, we evaluate perplexity on the word-level [WikiText-103 test dataset](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip), and appropriately compute perplexity given the change in tokens when using our subword tokenizer.

We use the following command to run WikiText-103 evaluation on a 345M parameter model.
<pre>
TASK="WIKITEXT103"

VALID_DATA=&#60;wikitext path&#62;.txt
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m

COMMON_TASK_ARGS="--num-layers 24 \
                  --hidden-size 1024 \
                  --num-attention-heads 16 \
                  --seq-length 1024 \
                  --max-position-embeddings 1024 \
                  --fp16 \
                  --vocab-file $VOCAB_FILE"

python tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file $MERGE_FILE \
       --load $CHECKPOINT_PATH \
       --batch-size 8 \
       --checkpoint-activations \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng
</pre>


<a id="lambada-cloze-accuracy"></a>
### LAMBADA Cloze Accuracy
To compute LAMBADA cloze accuracy (the accuracy of predicting the last token given the preceeding tokens) we utilize a detokenized, processed version of the [LAMBADA dataset](https://github.com/cybertronai/bflm/blob/master/lambada_test.jsonl).

We use the following command to run LAMBADA evaluation on a 345M parameter model. Note that the `--strict-lambada` flag should be used to require whole word matching. Make that `lambada` is part of the file path.

<pre>
TASK="LAMBADA"

VALID_DATA=&#60;lambada path&#62;.json
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m
COMMON_TASK_ARGS=&#60;same as those in <a href="#wikitext-perplexity-evaluation">WikiText Perplexity Evaluation</a> above&#62;

python tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type GPT2BPETokenizer \
       --strict-lambada \
       --merge-file $MERGE_FILE \
       --load $CHECKPOINT_PATH \
       --batch-size 8 \
       --checkpoint-activations \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng
</pre>

Further command line arguments are described in the source file [`main.py`](./tasks/main.py)

<a id="bert-task-evaluation"></a>
## BERT Task Evaluation
<a id="race-evaluation"></a>
### RACE Evaluation
The following script finetunes the BERT model for evaluation on the [RACE dataset](http://www.cs.cmu.edu/~glai1/data/race/). The `TRAIN_DATA` and `VALID_DATA` directory contain the RACE dataset as separate `.txt` files.

<pre>
TRAIN_DATA="data/RACE/train/middle"
VALID_DATA="data/RACE/dev/middle \
            data/RACE/dev/high"
VOCAB_FILE=bert-vocab.txt
PRETRAINED_CHECKPOINT=checkpoints/bert_345m
CHECKPOINT_PATH=checkpoints/bert_345m_race
COMMON_TASK_ARGS="--num-layers 24 \
                  --hidden-size 1024 \
                  --num-attention-heads 16 \
                  --seq-length 512 \
                  --max-position-embeddings 512 \
                  --fp16 \
                  --vocab-file $VOCAB_FILE"

COMMON_TASK_ARGS_EXT="--train-data $TRAIN_DATA \
                      --valid-data $VALID_DATA \
                      --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
                      --checkpoint-activations \
                      --save-interval 10000 \
                      --save $CHECKPOINT_PATH \
                      --log-interval 100 \
                      --eval-interval 1000 \
                      --eval-iters 10 \
                      --weight-decay 1.0e-1"

python tasks/main.py \
       --task RACE \
       $COMMON_TASK_ARGS \
       $COMMON_TASK_ARGS_EXT \
       --tokenizer-type BertWordPieceLowerCase \
       --epochs 3 \
       --batch-size 4 \
       --lr 1.0e-5 \
       --warmup 0.06
</pre>

<a id="mnli-evaluation"></a>
### MNLI Evaluation
The following script finetunes the BERT model for evaluation with the [MultiNLI sentence pair corpus](https://www.nyu.edu/projects/bowman/multinli/). Because the matching tasks are quite similar, the script can be quickly tweaked to work with the [Quora Question Pairs](https://www.kaggle.com/quora/question-pairs-dataset) (QQP) dataset as well.

<pre>

TRAIN_DATA="data/glue_data/MNLI/train.tsv"
VALID_DATA="data/glue_data/MNLI/dev_matched.tsv \
            data/glue_data/MNLI/dev_mismatched.tsv"
PRETRAINED_CHECKPOINT=checkpoints/bert_345m
VOCAB_FILE=bert-vocab.txt
CHECKPOINT_PATH=checkpoints/bert_345m_mnli
COMMON_TASK_ARGS=&#60;same as those in <a href="#race-evaluation">RACE Evaluation</a> above&#62;
COMMON_TASK_ARGS_EXT=&#60;same as those in <a href="#race-evaluation">RACE Evaluation</a> above&#62;

python tasks/main.py \
       --task MNLI \
       $COMMON_TASK_ARGS \
       $COMMON_TASK_ARGS_EXT \
       --tokenizer-type BertWordPieceLowerCase \
       --epochs 5 \
       --batch-size 8 \
       --lr 5.0e-5 \
       --warmup 0.065
</pre>

<a id="datasets"></a>
# Datasets
We do not host any datasets for GPT-2 or BERT training, however, we detail their collection so that our results may be reproduced.

<a id="collecting-wikipedia-training-data"></a>
## Collecting Wikipedia Training Data
We recommend following the Wikipedia data extraction process specified by Google research: "the recommended pre-processing is to download [the latest dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2), extract the text with [WikiExtractor.py](https://github.com/attardi/wikiextractor), and then apply any necessary cleanup to convert it into plain text."

We recommend using the `--json` argument when using WikiExtractor, which will dump the Wikipedia data into loose json format (one json per line), making it more manageable on the file system and also readily consumable by our codebase. We recommend further preprocessing this json dataset by nltk punctuation standardization. For BERT training, add newlines between sentences during data preprocessing. This is done with the `--split-sentences` flag in `preprocess_data.py` as described [above](#data-preprocessing). (Note that if you'd like to use Wikipedia data for GPT-2 training you should still clean it with nltk/spacy/ftfy, but do not split it into newline separated sentences.)

<a id="collecting-gpt-2-webtext-data"></a>
## Collecting GPT-2 Webtext Data
We utilize the publicly available [OpenWebText](https://github.com/eukaryote31/openwebtext) library from [jcpeterson](https://github.com/jcpeterson/openwebtext) and [eukaryote31's](https://github.com/eukaryote31/openwebtext) work to download urls. We then filtered, cleaned, and deduplicated all downloaded content according to the procedure described in our [openwebtext](./tools/openwebtext) directory. For reddit URLs corresponding to content up to October 2018 we arrived at approximately 37GB of content.
