import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import warnings
from transformers import logging
from typing import Optional, List, Union

# 配置静默模式
warnings.filterwarnings("ignore")
logging.set_verbosity_error()


class LocalLLM:
    def __init__(self, model_path: str = "/model/LLM/DeepSeek-V2-Lite-Chat"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """模型加载方法"""
        print(f"🚀 Loading {self.model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            print("✅ Model load successful！ (FP16)")
        except Exception as e:
            raise RuntimeError(f"❌ Load failed: {str(e)}")

    def generate(
            self,
            prompt: str,
            max_tokens: int = 600,
            temperature: float = 0.5,
            top_p: float = 0.9,
            stop: Optional[Union[str, List[str]]] = None
    ) -> str:
        """
        生成文本
        Args:
            prompt: 输入文本
            max_tokens: 最大生成token数
            temperature: 控制随机性 (0.1-1.0)
            top_p: 核采样阈值 (0.5-0.95)
            stop: 停止词 (str或list), 如 "\n" 或 ["###", "</s>"]
        """
        # 参数检查
        assert 0.1 <= temperature <= 1.0, "temperature should be in [0.1, 1.0]"
        assert 0.5 <= top_p <= 0.95, "top_p should be in [0.5, 0.95]"
        if stop is not None:
            stop = [stop] if isinstance(stop, str) else list(stop)

        # 构建输入
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt"
        ).to(self.model.device)

        # 定义停止条件类
        class StopOnTokens(StoppingCriteria):
            def __init__(self, tokenizer, stop_words):
                self.tokenizer = tokenizer
                # 预编码所有停止词（处理多token情况）
                self.stop_token_sequences = [tokenizer.encode(stop, add_special_tokens=False) for stop in stop_words]
                self.max_stop_len = max(len(seq) for seq in self.stop_token_sequences) if stop_words else 0

            def __call__(self, input_ids, scores, **kwargs):
                # 检查最后N个token（N=最长停止词长度+缓冲）
                check_len = min(32, self.max_stop_len + 4)  # 最多检查32个token
                recent_tokens = input_ids[0][-check_len:].tolist()

                # 检查所有停止词序列
                for stop_seq in self.stop_token_sequences:
                    if len(stop_seq) > len(recent_tokens):
                        continue
                    if recent_tokens[-len(stop_seq):] == stop_seq:
                        return True
                return False

        # 生成配置
        generate_args = {
            "input_ids": inputs,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0.1,
            "pad_token_id": self.tokenizer.eos_token_id,
            "stopping_criteria": StoppingCriteriaList([StopOnTokens(self.tokenizer, stop)]) if stop else None
        }

        # 执行生成
        outputs = self.model.generate(**generate_args)
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

        # 后处理停止词（确保兼容性）
        if stop:
            for stop_word in stop:
                response = response.split(stop_word)[0]
        return response.strip()


# 单例服务
llm_service = LocalLLM()


def get_response(
        prompt: str,
        max_tokens: int = 600,
        temperature: float = 0.5,
        top_p: float = 0.9,
        stop: Optional[Union[str, List[str]]] = None
) -> str:
    """
    获取模型响应
    Args:
        stop: 支持字符串或列表格式，例如：
              - stop="\n"          # 遇到换行符停止
              - stop=["###", "</s>"] # 遇到任意停止词停止
    """
    return llm_service.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop
    )


if __name__ == "__main__":
    response = get_response("中文解释一下Text2SQL的含义")
    print(get_response("杭州美食", stop="鱼"))

    # 遇到换行停止
    print(get_response("写一首诗", stop="\n"))
    print("\n💬 Response:")
    print(response)
