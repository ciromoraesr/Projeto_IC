import hashlib
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import tiktoken
from dotenv import load_dotenv
from groq import Groq
from rouge_score import rouge_scorer

load_dotenv()


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


API_KEY = os.getenv("API_KEY_ENV")
MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.1
MIN_INPUT_TOKENS = 5
MIN_API_TOKENS = 20
MIN_LLM_PERCENT_SAVINGS = 0.02
MIN_COMPLETION_TOKENS = 32
MAX_COMPLETION_TOKENS = 512
API_TIMEOUT = 30

SYSTEM_PROMPT_ANALYZE = (
    "You are an advanced text analysis engine. Your job is to analyze the user's prompt "
    "and create a strict structural optimization guide for a secondary compression engine.\n\n"
    "Identify specific verbose phrasing, conversational filler, or redundancies that can be safely deleted. "
    "Identify core operational requirements, variables, constraints, or tones that MUST be preserved.\n\n"
    "Format your output exactly like this with no other commentary:\n"
    "[REDUNDANCIES]\n"
    "- <verbose phrase or filler to delete>\n"
    "- <next verbose phrase to delete>\n\n"
    "[CORE_PRESERVATIONS]\n"
    "- <critical constraint or context item to keep>\n"
    "- <next critical constraint to keep>"
)

SYSTEM_PROMPT_COMPRESS = (
    "You are a ruthless prompt optimization engine. You will be provided with an <original_prompt> "
    "and a structural <analysis_guide> detailing precisely what to delete and what to save.\n\n"
    "Your objective is to rewrite the prompt into its absolute maximum token density using an ultra-concise, "
    "telegraphic style (strip unnecessary articles like 'a', 'an', 'the' where safe).\n\n"
    "Rules:\n"
    "1. Completely strip out every redundant item and phrase flagged in the [REDUNDANCIES] guide.\n"
    "2. Ensure all operational directives and constraints flagged under [CORE_PRESERVATIONS] are fully maintained.\n"
    "3. Retain 100% of the technical context and intent.\n"
    "4. Output ONLY the resulting compressed prompt text. Do not wrap it in tags, do not provide an explanation, "
    "and do not include any preamble."
)


@dataclass
class CompressionAnalysis:
   
    redundancies: List[str]
    verbosity_score: float  
    key_preservations: List[str]  


@dataclass
class RougeScores:

    rouge1: Dict[str, float]
    rouge2: Dict[str, float]
    rougeL: Dict[str, float]


class PromptCompressor:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or API_KEY
        if not self.api_key:
            raise ValueError("API key not set. Please set the API_KEY_ENV environment variable.")

        self.client = Groq(api_key=self.api_key)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.cache: Dict[str, dict] = {}
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )

    def count_tokens(self, text: str) -> int:
        
        return len(self.tokenizer.encode(text))

    def _normalize_prompt(self, prompt: str) -> str:
    
        text = prompt.replace("\r\n", "\n").replace("\r", "\n")
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        deduped_lines = []
        previous_line = None
        for line in text.split("\n"):
            if line == previous_line and line.strip():
                continue
            deduped_lines.append(line)
            previous_line = line

        return "\n".join(deduped_lines).strip()

    def _cache_key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _dynamic_max_tokens(self, source_tokens: int) -> int:
        budget = max(MIN_COMPLETION_TOKENS, int(source_tokens * 0.6))
        return min(MAX_COMPLETION_TOKENS, budget)

    def _looks_like_explanation(self, text: str) -> bool:
        
        lowered = text.lower()
        explanation_markers = (
            "here's", "here is", "explanation", "compressed prompt:",
            "i compressed", "changes made", "sure,", "certainly,",
            "here are", "let me", "original prompt",
        )
        if any(marker in lowered for marker in explanation_markers):
            return True

        answer_indicators = (
            "|---|", "your answer", "your request", "in response",
            "def ", "import ", "class ",
        )
        if any(indicator in lowered for indicator in answer_indicators):
            return True

        return False

    def _run_structural_analysis(self, prompt: str) -> Tuple[str, CompressionAnalysis]:
        
        try:
            logger.info("Stage 1: Analyzing target prompt for systemic redundancy...")
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_ANALYZE},
                    {"role": "user", "content": prompt},
                ],
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=256,
                top_p=1,
                timeout=API_TIMEOUT,
            )
        except Exception as e:
            logger.warning(f"Analysis engine failed, bypassing Stage 1 safely: {str(e)}")
            return "", CompressionAnalysis([], 0.0, [])

        analysis_text = (chat_completion.choices[0].message.content or "").strip()
        
        
        redundancies = []
        key_preservations = []
        current_section = None
        
        for line in analysis_text.split("\n"):
            line = line.strip()
            if "[REDUNDANCIES]" in line:
                current_section = "red"
                continue
            elif "[CORE_PRESERVATIONS]" in line:
                current_section = "pres"
                continue
                
            if line.startswith("-") or line.startswith("•"):
                clean_item = line.lstrip("-• ").strip()
                if current_section == "red":
                    redundancies.append(clean_item)
                elif current_section == "pres":
                    key_preservations.append(clean_item)

        verbosity_score = min(1.0, len(redundancies) / 5.0) if redundancies else 0.0

        analysis_obj = CompressionAnalysis(
            redundancies=redundancies,
            verbosity_score=round(verbosity_score, 2),
            key_preservations=key_preservations
        )
        
        return analysis_text, analysis_obj

    def calculate_rouge_scores(self, original: str, compressed: str) -> RougeScores:
        
        original_clean = original.strip() if original else ""
        compressed_clean = compressed.strip() if compressed else ""
        
        if not original_clean or not compressed_clean:
            return RougeScores(
                rouge1={'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                rouge2={'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                rougeL={'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
            )
        
        try:
            scores = self.rouge_scorer.score(original_clean, compressed_clean)
            return RougeScores(
                rouge1={k: round(getattr(scores['rouge1'], k), 4) for k in ['precision', 'recall', 'fmeasure']},
                rouge2={k: round(getattr(scores['rouge2'], k), 4) for k in ['precision', 'recall', 'fmeasure']},
                rougeL={k: round(getattr(scores['rougeL'], k), 4) for k in ['precision', 'recall', 'fmeasure']}
            )
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {str(e)}")
            return RougeScores(
                rouge1={'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                rouge2={'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                rougeL={'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
            )

    def evaluate_compression_quality(self, results: List[dict]) -> dict:
        
        all_scores = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'ratios': [], 'savings': []}
        
        for res in results:
            rouge = self.calculate_rouge_scores(res['original'], res['compressed'])
            all_scores['rouge1'].append(rouge.rouge1['fmeasure'])
            all_scores['rouge2'].append(rouge.rouge2['fmeasure'])
            all_scores['rougeL'].append(rouge.rougeL['fmeasure'])
            
            o_len, c_len = len(res['original'].split()), len(res['compressed'].split())
            all_scores['ratios'].append(c_len / o_len if o_len > 0 else 1.0)
            all_scores['savings'].append(res['reduction_percentage'])
        
        n = len(results)
        return {
            'rouge1_avg': round(sum(all_scores['rouge1']) / n, 4),
            'rouge2_avg': round(sum(all_scores['rouge2']) / n, 4),
            'rougeL_avg': round(sum(all_scores['rougeL']) / n, 4),
            'avg_compression_ratio': round(sum(all_scores['ratios']) / n, 4),
            'avg_token_savings': round(sum(all_scores['savings']) / n, 2),
            'num_evaluations': n,
        }

    def _build_result(
        self, original: str, compressed: str, original_tokens: int,
        compressed_tokens: int, strategy: str, api_usage_stats: Optional[dict] = None,
        analysis: Optional[CompressionAnalysis] = None,
    ) -> dict:
       
        token_reduction = original_tokens - compressed_tokens
        reduction_percentage = (token_reduction / original_tokens * 100) if original_tokens > 0 else 0
        rouge_scores = self.calculate_rouge_scores(original, compressed)

        result = {
            "original": original,
            "compressed": compressed,
            "original_local_tokens": original_tokens,
            "compressed_local_tokens": compressed_tokens,
            "token_reduction": token_reduction,
            "reduction_percentage": round(reduction_percentage, 2),
            "strategy": strategy,
            "api_usage_stats": api_usage_stats or {"prompt": 0, "completion": 0, "total": 0},
            "rouge_scores": {
                "rouge1": rouge_scores.rouge1,
                "rouge2": rouge_scores.rouge2,
                "rougeL": rouge_scores.rougeL,
            }
        }
        if analysis:
            result["analysis"] = {
                "redundancies": analysis.redundancies,
                "verbosity_score": analysis.verbosity_score,
                "key_preservations": analysis.key_preservations,
            }
        return result

    def compress(self, prompt: str) -> dict:
      
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        original = prompt.strip()
        original_tokens = self.count_tokens(original)

        if original_tokens < MIN_INPUT_TOKENS:
            raise ValueError(f"Prompt too short to optimize ({original_tokens} tokens)")

        normalized = self._normalize_prompt(original)
        cache_key = self._cache_key(normalized)
        if cache_key in self.cache:
            logger.info("Cache hit verified.")
            return self.cache[cache_key]

        normalized_tokens = self.count_tokens(normalized)

        analysis_guide_text, analysis_data = self._run_structural_analysis(normalized)

      
        max_tokens = self._dynamic_max_tokens(normalized_tokens)
        
    
        execution_payload = (
            f"<original_prompt>\n{normalized}\n</original_prompt>\n\n"
            f"<analysis_guide>\n{analysis_guide_text}\n</analysis_guide>"
        )

        try:
            logger.info(f"Stage 2: Executing guided surgical extraction ({normalized_tokens} baseline tokens)...")
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_COMPRESS},
                    {"role": "user", "content": execution_payload},
                ],
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=max_tokens,
                top_p=1,
                timeout=API_TIMEOUT,
            )
        except Exception as e:
            logger.error(f"API execution link collapsed: {str(e)}")
            raise Exception(f"API execution failed: {str(e)}")

        compressed = (chat_completion.choices[0].message.content or "").strip()

        compressed = re.sub(r"^<compressed_prompt>\n?|\n?</compressed_prompt>$", "", compressed)
        compressed_tokens = self.count_tokens(compressed) if compressed else 0

        llm_percent_savings = ((normalized_tokens - compressed_tokens) / normalized_tokens) if normalized_tokens else 0
        llm_result_is_valid = (
            bool(compressed)
            and not self._looks_like_explanation(compressed)
            and compressed.lower() != normalized.lower()
            and compressed_tokens < normalized_tokens
            and llm_percent_savings >= MIN_LLM_PERCENT_SAVINGS
        )

        if llm_result_is_valid:
            logger.info(f"Surgical optimization verified: {normalized_tokens} → {compressed_tokens} tokens.")
            result = self._build_result(
                original=original, compressed=compressed,
                original_tokens=original_tokens, compressed_tokens=compressed_tokens,
                strategy="two_stage_pipeline",
                api_usage_stats={
                    "prompt": chat_completion.usage.prompt_tokens,
                    "completion": chat_completion.usage.completion_tokens,
                    "total": chat_completion.usage.total_tokens,
                },
                analysis=analysis_data,
            )
        elif normalized_tokens < original_tokens:
            logger.info("Pipeline optimization marginal. Returning normalized variant.")
            result = self._build_result(
                original=original, compressed=normalized,
                original_tokens=original_tokens, compressed_tokens=normalized_tokens,
                strategy="local_normalization", analysis=analysis_data,
            )
        else:
            logger.info("String is already fully dense. Retaining original instance safely.")
            result = self._build_result(
                original=original, compressed=original,
                original_tokens=original_tokens, compressed_tokens=original_tokens,
                strategy="uncompressed_fallback", analysis=analysis_data,
            )

        self.cache[cache_key] = result
        return result


def print_rouge_scores(rouge_scores: Dict) -> None:
    print(f"\n{'='*60}\nROUGE SCORES:\n{'='*60}")
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        scores = rouge_scores[metric]
        print(f"\n{metric.upper()}:\n  Precision: {scores['precision']}\n  Recall:    {scores['recall']}\n  F-measure: {scores['fmeasure']}")


def print_batch_evaluation(eval_results: Dict) -> None:
    print(f"\n{'='*60}\nBATCH EVALUATION RESULTS:\n{'='*60}")
    print(f"Number of evaluations: {eval_results['num_evaluations']}")
    print(f"\nAverage ROUGE Scores:\n  ROUGE-1 F-measure: {eval_results['rouge1_avg']}\n  ROUGE-2 F-measure: {eval_results['rouge2_avg']}\n  ROUGE-L F-measure: {eval_results['rougeL_avg']}")
    print(f"\nCompression Metrics:\n  Avg compression ratio: {eval_results['avg_compression_ratio']}\n  Avg token savings: {eval_results['avg_token_savings']}%")


if __name__ == "__main__":
    try:
        compressor = PromptCompressor()

        problematic_string = (
            "Write a comprehensive guide explaining quantum computing in detail, including its "
            "fundamental principles, current real-world applications, technical challenges, and "
            "future prospects and possibilities. The guide should be suitable for someone with a "
            "basic understanding of physics. Please make it detailed and thorough."
        )

        num_runs = 30
       
        original_tokens_list = []
        compressed_tokens_list = []
        token_reduction_list = []
        reduction_percentage_list = []
        rouge1_f_list = []
        rouge2_f_list = []
        rougeL_f_list = []
        verbosity_scores = []
        num_redundancies_list = []
        strategies = []

        def get_rouge_f(rouge_dict, metric_name_variants):
            """
            Safely extract a ROUGE F-measure (or raw score) for a given metric.
            metric_name_variants: list of possible key names, e.g. ['rouge-1', 'rouge1', 'rouge_1']
            Returns float score.
            """
            for variant in metric_name_variants:
                if variant in rouge_dict:
                    val = rouge_dict[variant]
                    if isinstance(val, (int, float)):
                        return float(val)
                    elif isinstance(val, dict):
                      
                        if 'f' in val:
                            return float(val['f'])
                       
                        for subkey in ['fmeasure', 'f1', 'score']:
                            if subkey in val:
                                return float(val[subkey])
                       
                        for v in val.values():
                            if isinstance(v, (int, float)):
                                return float(v)
                   
            base = metric_name_variants[0].replace('-', '').replace('_', '').lower()
            for key in rouge_dict:
                if base in key.lower().replace('-', '').replace('_', ''):
                    val = rouge_dict[key]
                    if isinstance(val, (int, float)):
                        return float(val)
                    elif isinstance(val, dict):
                        if 'f' in val:
                            return float(val['f'])
                     
            return 0.0

        for i in range(num_runs):
            result = compressor.compress(problematic_string)

           
            orig_tok = result['original_local_tokens']
            comp_tok = result['compressed_local_tokens']
            original_tokens_list.append(orig_tok)
            compressed_tokens_list.append(comp_tok)

            token_reduction_list.append(result['token_reduction'])

          
            red_pct = result['reduction_percentage']
            if isinstance(red_pct, str):
                red_pct = float(red_pct.replace('%', '').strip())
            reduction_percentage_list.append(float(red_pct))

         
            rouge = result['rouge_scores']
            rouge1_f_list.append(get_rouge_f(rouge, ['rouge-1', 'rouge1', 'rouge_1', 'rouge_1']))
            rouge2_f_list.append(get_rouge_f(rouge, ['rouge-2', 'rouge2', 'rouge_2', 'rouge_2']))
            rougeL_f_list.append(get_rouge_f(rouge, ['rouge-l', 'rougeL', 'rouge_L', 'rougel', 'rouge_l']))

           
            if 'analysis' in result:
                analysis = result['analysis']
                verbosity_scores.append(analysis['verbosity_score'])
                num_redundancies_list.append(len(analysis['redundancies']))
            else:
                verbosity_scores.append(0.0)
                num_redundancies_list.append(0)

            strategies.append(result['strategy'])

        def mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        avg_orig_tok = mean(original_tokens_list)
        avg_comp_tok = mean(compressed_tokens_list)
        avg_token_reduction_abs = mean(token_reduction_list)
        avg_reduction_pct = mean(reduction_percentage_list)
        avg_rouge1 = mean(rouge1_f_list)
        avg_rouge2 = mean(rouge2_f_list)
        avg_rougeL = mean(rougeL_f_list)
        avg_verbosity = mean(verbosity_scores)
        avg_redundancies = mean(num_redundancies_list)

        from collections import Counter
        strategy_counts = Counter(strategies)
        most_common_strategy = strategy_counts.most_common(1)[0][0] if strategy_counts else "N/A"

        print(f"\n{'='*60}\nAVERAGE RESULTS OVER {num_runs} RUNS\n{'='*60}")
        print(f"Average original tokens:         {avg_orig_tok:.2f}")
        print(f"Average compressed tokens:       {avg_comp_tok:.2f}")
        print(f"Average token reduction (abs):   {avg_token_reduction_abs:.2f}")
        print(f"Average reduction percentage:    {avg_reduction_pct:.2f}%")
        print(f"Average ROUGE-1:                 {avg_rouge1:.4f}")
        print(f"Average ROUGE-2:                 {avg_rouge2:.4f}")
        print(f"Average ROUGE-L:                 {avg_rougeL:.4f}")
        print(f"Average verbosity score:         {avg_verbosity:.4f} / 1.0")
        print(f"Average redundancies removed:    {avg_redundancies:.2f}")
        print(f"Most deployed strategy:          {most_common_strategy}")
        print(f"{'='*60}")

    except Exception as e:
        logger.error(f"Execution Error: {e}")
        sys.exit(1)
