import json
import os
import re
import subprocess
import sys
import torch
from pathlib import Path

# GPU 选择:默认用环境变量 CUDA_VISIBLE_DEVICES,若未设置则用全部可用 GPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"))

# 离线加载(本机访问不了 huggingface.co,已从 hf-mirror 补全缓存)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
from llamafactory.data.template import TEMPLATES
from llamafactory.data.template import get_template_and_fix_tokenizer

# 与训练保持一致:llamafactory 训练时用 AutoModelForImageTextToText 加载
# Qwen3.5-9B(config.architectures 是 Qwen3_5ForConditionalGeneration,匹配该映射)
# 用 AutoModelForCausalLM 加载会导致层名不匹配、LoRA adapter 挂不上
base_model_name = "Qwen/Qwen3.5-9B"
adapter_path = "./toolfineturn/saves/qwen3.5-9b/lora/sft/checkpoint-105"
# 训练用的对话模板(与训练一致,不用模型自带的带 think 模板)
TEMPLATE_NAME = "qwen3_5_nothink"

WORKDIR = Path.cwd()
SYSTEM = "You are a helpful assistant solve problem with tools"

# ---------- 工具函数 ----------
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_read(file_path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(file_path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

def run_edit(file_path: str, old_string: str, new_string: str) -> str:
    try:
        path = safe_path(file_path)
        text = path.read_text()
        if old_string not in text:
            return f"Error: text not found in {file_path}"
        path.write_text(text.replace(old_string, new_string, 1))
        return f"Edited {file_path}"
    except Exception as e:
        return f"Error: {e}"

def run_glob(pattern: str) -> str:
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"

# ---------- 工具定义(OpenAI 格式;传给 apply_chat_template,渲染方式与训练完全一致) ----------
# 注意:训练数据中的工具名为 bash / read / edit,参数为 command / file_path / old_string / new_string。
# 这里保持与训练一致,并额外提供 write_file / glob 作为补充能力。
TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read", "description": "Read file contents.",
        "parameters": {"type": "object",
                       "properties": {"file_path": {"type": "string"},
                                      "limit": {"type": "integer"}},
                       "required": ["file_path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write content to a file.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit", "description": "Replace exact text in a file once.",
        "parameters": {"type": "object",
                       "properties": {"file_path": {"type": "string"},
                                      "old_string": {"type": "string"},
                                      "new_string": {"type": "string"}},
                       "required": ["file_path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "glob", "description": "Find files matching a glob pattern.",
        "parameters": {"type": "object",
                       "properties": {"pattern": {"type": "string"}},
                       "required": ["pattern"]}}},
]

TOOL_HANDLERS = {
    "bash": run_bash,
    "read": run_read,
    "write_file": run_write,
    "edit": run_edit,
    "glob": run_glob,
}

# ---------- 解析工具调用(<tool_call> XML 风格,与 Qwen3 模板渲染一致) ----------
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
FUNC_RE = re.compile(r"<function=([^>]+)>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.S)

def parse_tool_calls(text: str) -> list[dict]:
    """从 assistant 输出提取 <tool_call> 块 → [{name, arguments(JSON字符串)}]"""
    calls = []
    for block in TOOL_CALL_RE.findall(text):
        m = FUNC_RE.search(block)
        if not m:
            continue
        name = m.group(1).strip()
        args = {k.strip(): v.strip() for k, v in PARAM_RE.findall(block)}
        calls.append({"name": name, "arguments": json.dumps(args, ensure_ascii=False)})
    return calls

def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.replace("<think>", "").replace("</think>", "")

# ---------- agent loop(消息用 OpenAI 格式存,每轮用 llamafactory 模板 + tools 渲染) ----------
# 关键点:训练时 llamafactory 用 qwen3_5_nothink 模板渲染(无 <think>),
# 推理也必须用同一个模板,否则模型输出 <think> 块,行为与训练不一致。
def agent_loop(messages: list[dict], max_rounds: int = 10):
    for _ in range(max_rounds):
        # 用 llamafactory 模板渲染 prompt(与训练一致)
        # system 消息要单独传给 system= 参数,不能放在 messages 里
        system = next((m["content"] for m in messages if m["role"] == "system"), SYSTEM)
        render_messages = [m for m in messages if m["role"] != "system"] + [{"role": "assistant", "content": ""}]
        encoded = template._encode(tokenizer, render_messages, system=system, tools=json.dumps(TOOLS, ensure_ascii=False))
        prompt_ids = []
        for e in encoded[:-1]:
            prompt_ids += e
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            repetition_penalty=1.05,   # 防止退化成重复文本死循环
        )
        new_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=False)

        assistant_text = re.split(r"<\|im_end\|>|<\|endoftext\|>", raw)[0]

        tool_calls = parse_tool_calls(assistant_text)
        if not tool_calls:
            final_text = strip_thinking(
                tokenizer.decode(new_ids, skip_special_tokens=True).strip())
            print(final_text)
            messages.append({"role": "assistant", "content": final_text})
            return

        # 截到最后一个 </tool_call>,丢弃后面的杂音
        end = assistant_text.rfind("</tool_call>") + len("</tool_call>")
        call_part = assistant_text[:end]

        # 回填助手回合:用 llamafactory 的 function 角色,content 是工具调用 JSON
        # 模板会把它渲染回 <tool_call> XML,与训练数据一致(function_call 消息)
        calls = []
        for c in tool_calls:
            args_json = c["arguments"]
            call_json = json.dumps({"name": c["name"], "arguments": json.loads(args_json)}, ensure_ascii=False)
            calls.append(call_json)
        # 多个工具调用:并行调用,content 是 JSON 数组
        messages.append({"role": "function", "content": json.dumps([json.loads(c) for c in calls], ensure_ascii=False) if len(calls) > 1 else calls[0]})

        for call in tool_calls:
            name = call["name"]
            args = json.loads(call["arguments"])
            print(f"\033[33m> {name}\033[0m")
            handler = TOOL_HANDLERS.get(name)
            try:
                output = handler(**args) if handler else f"Unknown: {name}"
            except Exception as e:
                output = f"Error: {e}"
            print(str(output)[:200])
            # 工具结果按 observation 角色回填(与训练数据的 observation 消息一致)
            messages.append({"role": "observation", "content": str(output)})

    print("(reached max rounds)")

# ---------- 交互入口 ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3.5-9B + LoRA tool-use agent")
    parser.add_argument("queries", nargs="*", help="非交互模式:直接传入问题")
    parser.add_argument("--no-lora", action="store_true",
                        help="不加载 LoRA adapter,用基座模型(其原生工具调用能力更强)")
    parser.add_argument("--gpu", default="0,1,2,3",
                        help="指定使用的 GPU 编号(默认 0,1,2,3)")
    parser.add_argument("--adapter", default=adapter_path,
                        help=f"LoRA adapter 路径(默认 {adapter_path})")
    cli = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = cli.gpu

    print("Qwen3.5-9B + LoRA Tool Use(<tool_call> 格式)")
    print("输入问题,回车发送。输入 q 退出。\n")

    model = AutoModelForImageTextToText.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device_map="auto",
        trust_remote_code=True,
    )
    if not cli.no_lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cli.adapter)
        print(f"(已加载 LoRA: {cli.adapter})")
    else:
        print("(--no-lora: 未加载 LoRA,使用基座模型)")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 初始化 llamafactory 模板(与训练一致)
    template = TEMPLATES[TEMPLATE_NAME]

    # 多模态架构可能需要 processor,初始化以防后续用到
    try:
        processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    except Exception:
        processor = None

    messages = [{"role": "system", "content": SYSTEM}]   # 跨轮保留
    if cli.queries:
        # 非交互模式:处理命令行传入的问题后退出(便于测试)
        for query in cli.queries:
            print(f"\033[36magent >> \033[0m{query}")
            messages.append({"role": "user", "content": query})
            agent_loop(messages)
            print()
        raise SystemExit(0)
    while True:
        try:
            query = input("\033[36magent >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        messages.append({"role": "user", "content": query})
        agent_loop(messages)
        print()
