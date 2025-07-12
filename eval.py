from src.fileloader import dataloader

if __name__ == '__main__':
    dat= dataloader.datas("./data/OKVQA")
    input="/root/autodl-tmp/RoG/qwen/res/llama/output_roglora_results_ro2.jsonl"
    dat.evaluate_jsonl(input)