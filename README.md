# Court Opinion Annotation Project

## Contents
- `*.txt`: 20 collected New York Court of Appeals opinion/document texts
- `metadata.csv`: case metadata and source URLs
- `annotate_openrouter.py`: calls OpenRouter and saves successful annotations incrementally to `annotated_dataset.jsonl`
- `visualize_annotations.py`: creates `annotation_visualization.png` from successful annotations

## Run
1. Install dependencies:
   `python3 -m pip install requests matplotlib`
2. Set your OpenRouter API key in Terminal (do not put it in this folder or share it):
   `export OPENROUTER_API_KEY='YOUR_KEY'`
3. Run the annotation:
   `python3 annotate_openrouter.py`
4. If rate-limited, wait and rerun the same command. Successful records are reused, so it resumes rather than starting over.
5. When at least one document succeeds, create the visualization:
   `python3 visualize_annotations.py`

Default model: `google/gemma-4-31b-it:free`. Free endpoints can still impose provider/rate limits. This script prints the response details when OpenRouter rejects a request; do not repeatedly rerun if the same 429 persists.

## Outputs
- `annotated_dataset.jsonl`: one JSON record per successfully annotated document
- `annotation_errors.jsonl`: errors for documents that could not be annotated
- `annotation_visualization.png`: chart of court disposition and document type

## Annotation limitations
Model-generated labels are preliminary research annotations, not legal conclusions. Verify dispositions and summaries against the underlying opinions before relying on them.
