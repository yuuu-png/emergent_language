import wandb

from transformers import GPT2TokenizerFast
import torch

from train import TranslateModule, generate_square_subsequent_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
EOS_IDX = tokenizer.eos_token_id
BOS_IDX = tokenizer.bos_token_id


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, first_tokens=[]):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (
            generate_square_subsequent_mask(ys.size(0), DEVICE).type(torch.bool)
        ).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        if len(first_tokens) > 0:
            next_word = first_tokens.pop(0)

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


def main():
    apt = wandb.Api()
    artifact = apt.artifact("ishiyama-k/translate/model-0mzidsdc:v1")
    path = artifact.download()
    module = TranslateModule.load_from_checkpoint(f"{path}/model.chpt").eval()
    model = module.model

    for i in range(32):
        start_with = tokenizer.encode(f"{tokenizer.bos_token}This is a photo of a")[1:]

        src = torch.tensor(i)
        src_mask = (torch.zeros(512, 512)).type(torch.bool)
        tgt_tokens = greedy_decode(
            model,
            src,
            src_mask,
            max_len=100,
            start_symbol=BOS_IDX,
            first_tokens=start_with,
        ).flatten()
        print(f"{i} : {tokenizer.decode(tgt_tokens)}")


if __name__ == "__main__":
    main()
