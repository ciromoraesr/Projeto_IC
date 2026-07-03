import uuid
import statistics
from flask import Flask, abort, render_template, request, jsonify
from compressor import PromptCompressor

app = Flask(__name__)
compressor = PromptCompressor()
histories = {}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt = request.form.get("prompt")
        history_id = request.form.get("history_id")

        if prompt:
            try:
                result = compressor.compress(prompt)
                history = histories.get(history_id)

                if history is None:
                    history_id = str(uuid.uuid4())
                    history = {
                        "id": history_id,
                        "original_prompt": prompt,
                        "steps": [],
                        "rouge_history": [],  
                    }
                    histories[history_id] = history

                
                rouge_scores = result.get("rouge_scores", {})
                
                step_data = {
                    "step_number": len(history["steps"]) + 1,
                    "input_prompt": prompt,
                    "compressed_prompt": result["compressed"],
                    "original_tokens": result["original_local_tokens"],
                    "compressed_tokens": result["compressed_local_tokens"],
                    "token_reduction": result["token_reduction"],
                    "reduction_percentage": result["reduction_percentage"],
                    "strategy": result.get("strategy", "unknown"),
                    "analysis": result['analysis'] if "analysis" in result else None,
                    "rouge_scores": {
                        "rouge1": rouge_scores.get("rouge1", {}),
                        "rouge2": rouge_scores.get("rouge2", {}),
                        "rougeL": rouge_scores.get("rougeL", {}),
                    },
                    "api_usage": result.get("api_usage_stats", {"prompt": 0, "completion": 0, "total": 0}),
                }
                
                history["steps"].append(step_data)
                history["rouge_history"].append({
                    "step": len(history["steps"]),
                    "rouge1_fmeasure": rouge_scores.get("rouge1", {}).get("fmeasure", 0),
                    "rouge2_fmeasure": rouge_scores.get("rouge2", {}).get("fmeasure", 0),
                    "rougeL_fmeasure": rouge_scores.get("rougeL", {}).get("fmeasure", 0),
                })

                return render_template(
                    "result.html",
                    data=result,
                    history_id=history_id,
                    history_length=len(history["steps"]),
                    step_data=step_data,
                )
            except (ValueError, Exception) as e:
                error_message = str(e)
                return render_template("result.html", error=error_message)

    return render_template("index.html")


@app.route("/history/<history_id>")
def history(history_id):
    history_data = histories.get(history_id)
    if history_data is None:
        abort(404)


    if history_data["steps"]:
        avg_reduction = statistics.mean(
            [step["reduction_percentage"] for step in history_data["steps"]]
        )
        avg_rouge1 = statistics.mean(
            [step["rouge_scores"]["rouge1"].get("fmeasure", 0) 
             for step in history_data["steps"]]
        )
        avg_rouge2 = statistics.mean(
            [step["rouge_scores"]["rouge2"].get("fmeasure", 0) 
             for step in history_data["steps"]]
        )
        avg_rougeL = statistics.mean(
            [step["rouge_scores"]["rougeL"].get("fmeasure", 0) 
             for step in history_data["steps"]]
        )
        
        total_api_tokens = sum(
            [step["api_usage"].get("total", 0) for step in history_data["steps"]]
        )
        
        history_data["statistics"] = {
            "avg_reduction_percentage": round(avg_reduction, 2),
            "avg_rouge1_fmeasure": round(avg_rouge1, 4),
            "avg_rouge2_fmeasure": round(avg_rouge2, 4),
            "avg_rougeL_fmeasure": round(avg_rougeL, 4),
            "total_api_tokens_used": total_api_tokens,
            "num_compressions": len(history_data["steps"]),
        }

    return render_template("history.html", history=history_data)


@app.route("/api/history/<history_id>/stats", methods=["GET"])
def history_stats(history_id):

    history_data = histories.get(history_id)
    if history_data is None:
        abort(404)

    if not history_data["steps"]:
        return jsonify({
            "error": "No compression steps in this history"
        }), 400

    avg_reduction = statistics.mean(
        [step["reduction_percentage"] for step in history_data["steps"]]
    )
    avg_rouge1 = statistics.mean(
        [step["rouge_scores"]["rouge1"].get("fmeasure", 0) 
         for step in history_data["steps"]]
    )
    avg_rouge2 = statistics.mean(
        [step["rouge_scores"]["rouge2"].get("fmeasure", 0) 
         for step in history_data["steps"]]
    )
    avg_rougeL = statistics.mean(
        [step["rouge_scores"]["rougeL"].get("fmeasure", 0) 
         for step in history_data["steps"]]
    )
    
    total_api_tokens = sum(
        [step["api_usage"].get("total", 0) for step in history_data["steps"]]
    )
    
    return jsonify({
        "history_id": history_id,
        "num_compressions": len(history_data["steps"]),
        "avg_reduction_percentage": round(avg_reduction, 2),
        "avg_rouge_scores": {
            "rouge1_fmeasure": round(avg_rouge1, 4),
            "rouge2_fmeasure": round(avg_rouge2, 4),
            "rougeL_fmeasure": round(avg_rougeL, 4),
        },
        "total_api_tokens_used": total_api_tokens,
        "rouge_history": history_data["rouge_history"],
    })


@app.route("/api/compression/single", methods=["POST"])
def compression_api():

    data = request.get_json()
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        result = compressor.compress(prompt)
        
        return jsonify({
            "success": True,
            "original": result["original"],
            "compressed": result["compressed"],
            "original_tokens": result["original_local_tokens"],
            "compressed_tokens": result["compressed_local_tokens"],
            "token_reduction": result["token_reduction"],
            "reduction_percentage": result["reduction_percentage"],
            "strategy": result.get("strategy", "unknown"),
            "rouge_scores": result.get("rouge_scores", {}),
            "api_usage": result.get("api_usage_stats", {}),
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@app.route("/api/batch-evaluation", methods=["POST"])
def batch_evaluation():

    data = request.get_json()
    prompts = data.get("prompts", [])

    if not prompts or not isinstance(prompts, list):
        return jsonify({"error": "Please provide a list of prompts"}), 400

    results = []
    try:
        for prompt in prompts:
            if prompt and prompt.strip():
                result = compressor.compress(prompt)
                results.append({
                    "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                    "reduction_percentage": result["reduction_percentage"],
                    "strategy": result.get("strategy", "unknown"),
                    "rouge1_fmeasure": result.get("rouge_scores", {}).get("rouge1", {}).get("fmeasure", 0),
                    "rouge2_fmeasure": result.get("rouge_scores", {}).get("rouge2", {}).get("fmeasure", 0),
                    "rougeL_fmeasure": result.get("rouge_scores", {}).get("rougeL", {}).get("fmeasure", 0),
                })
        
        if not results:
            return jsonify({"error": "No valid prompts to evaluate"}), 400
        

        avg_reduction = statistics.mean([r["reduction_percentage"] for r in results])
        avg_rouge1 = statistics.mean([r["rouge1_fmeasure"] for r in results])
        avg_rouge2 = statistics.mean([r["rouge2_fmeasure"] for r in results])
        avg_rougeL = statistics.mean([r["rougeL_fmeasure"] for r in results])
        
        return jsonify({
            "success": True,
            "num_evaluations": len(results),
            "results": results,
            "batch_averages": {
                "avg_reduction_percentage": round(avg_reduction, 2),
                "avg_rouge1_fmeasure": round(avg_rouge1, 4),
                "avg_rouge2_fmeasure": round(avg_rouge2, 4),
                "avg_rougeL_fmeasure": round(avg_rougeL, 4),
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(error):
    return render_template("500.html"), 500


if __name__ == "__main__":
    app.run(debug=True)
