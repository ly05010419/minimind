# -*- coding: utf-8 -*-
"""
模型测试脚本
用法: python test_model.py --weight full_sft --hidden_size 512 --num_hidden_layers 8

结果保存到: test_results_{weight}_{hidden_size}x{num_hidden_layers}.txt
"""

import argparse
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
import warnings
import time
import os

warnings.filterwarnings('ignore')

def load_questions(filepath='question100.txt'):
    """读取问题文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return prompts

def test_model(args):
    """测试模型"""
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained('model')
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers
    )
    model = MiniMindForCausalLM(config)

    # 构建权重路径
    weight_path = f'./out/{args.weight}_{args.hidden_size}.pth'
    if not os.path.exists(weight_path):
        print(f'Error: Weight file not found: {weight_path}')
        return

    model.load_state_dict(torch.load(weight_path, map_location=args.device))
    model = model.eval().to(args.device)

    # 计算参数量
    param_count = sum(p.numel() for p in model.parameters()) / 1e6

    # 加载问题
    prompts = load_questions(args.question_file)

    print(f'Model: {args.weight} ({args.hidden_size}x{args.num_hidden_layers}, {param_count:.1f}M params)')
    print(f'Testing {len(prompts)} questions...')

    results = []
    start_time = time.time()

    for i, prompt in enumerate(prompts, 1):
        conversation = [{'role': 'user', 'content': prompt}]
        inputs = tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(inputs, return_tensors='pt').to(args.device)

        with torch.no_grad():
            output = model.generate(
                inputs['input_ids'],
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        results.append((prompt, response))

        if i % 20 == 0:
            print(f'  Progress: {i}/{len(prompts)}')

    total_time = time.time() - start_time

    # 保存结果
    output_file = f'test_results_{args.weight}_{args.hidden_size}x{args.num_hidden_layers}.txt'

    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入头部信息
        f.write('=' * 60 + '\n')
        f.write(f'Model: {args.weight}\n')
        f.write(f'Architecture: {args.hidden_size} x {args.num_hidden_layers}\n')
        f.write(f'Parameters: {param_count:.1f}M\n')
        f.write(f'Total Time: {total_time:.1f}s\n')
        f.write(f'Questions: {len(prompts)}\n')
        f.write('=' * 60 + '\n\n')

        # 写入每个问答
        for i, (prompt, response) in enumerate(results, 1):
            f.write(f'Q{i}: {prompt}\n')
            f.write(f'A: {response}\n')
            f.write('-' * 40 + '\n\n')

    print(f'\nDone! Results saved to: {output_file}')
    print(f'Total time: {total_time:.1f}s ({total_time/len(prompts):.2f}s per question)')

    return output_file

def main():
    parser = argparse.ArgumentParser(description='模型测试脚本')
    parser.add_argument('--weight', default='full_sft', type=str, help='权重名称')
    parser.add_argument('--hidden_size', default=512, type=int, help='隐藏层维度')
    parser.add_argument('--num_hidden_layers', default=8, type=int, help='层数')
    parser.add_argument('--max_tokens', default=150, type=int, help='最大生成长度')
    parser.add_argument('--temperature', default=0.85, type=float, help='温度')
    parser.add_argument('--top_p', default=0.85, type=float, help='top_p')
    parser.add_argument('--repetition_penalty', default=1.2, type=float, help='重复惩罚')
    parser.add_argument('--question_file', default='question100.txt', type=str, help='问题文件')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)

    args = parser.parse_args()
    test_model(args)

if __name__ == '__main__':
    main()
