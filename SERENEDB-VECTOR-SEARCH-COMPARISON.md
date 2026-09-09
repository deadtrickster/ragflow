# How the doc-store backends express vector search, and where SereneDB diverges

Written 2026-09-09 on lubuntu3, against a 42,801,989-chunk arXiv corpus (bge-m3, 1024-dim).
All four backends implement the same `DocStoreConnection` interface and receive the same
`MatchDenseExpr`, carrying `topn` and `extra_options = {"similarity": …, "num_candidates": …}`.

`rag/nlp/search.py` builds it with `get_vector(..., top_k=1024, num_candidates=..., similarity=0.1)`
and the retrieval layer above applies `similarity_threshold=0.2` **again**, in Python, after
optionally reranking `rerank_candidates_count=64` with a cross-encoder. So the SQL/DSL layer is a
**candidate generator**, not the final ranking.

## What each backend does with `similarity`

| backend | how the vector search is issued | where `similarity` is applied | recall knob |
|---|---|---|---|
| **es_conn** | `s.knn(field, k, num_candidates, query_vector, filter, similarity=)` | parameter to the kNN call — ES post-filters the k nearest | `num_candidates`, default `min(k*2, 10000)` |
| **opensearch_conn** | `knn_query[col] = {"vector": …, "k": topn}` | not passed into the kNN clause | `k` |
| **infinity_conn** | `MATCH_DENSE` with `extra_options` | renamed `similarity` → `threshold`, handed to the engine natively | engine-side |
| **serenedb_conn** | plain SQL against the index relation | ~~**compiled into the WHERE clause**~~ **FIXED 2026-09-09** — no longer applied in SQL | `sdb_ivf_search_nprobe`; `num_candidates` still dropped |

The first three all say: *"return the k approximate nearest neighbours, then discard those below
`similarity`."* The threshold never changes what the index searches — it filters the result.

SereneDB's translation says something different: *"return everything whose distance is within
`threshold`, then take k."* That is a **radius query, not a top-k query**, and the planner honours it
literally.

## What that costs

`EXPLAIN` of the RAGFlow-shaped query on 42.5M rows:

```
IRESEARCH_SCAN
  Index Filter:
    Vector Range          <-- radius search
      Field: q_1024_vec_n
      Metric: ip
      Radius: <= -0
    Term: kb_id = '5ba3ee10…'
  ~8,567,943 rows
```

With `similarity = 0.0` (and `0.1` is barely different for normalised embeddings) the radius admits
roughly every vector with non-negative inner product — ~8.5M of 42.5M — all enumerated so that
`TOP_N` can keep 10. Measured, four interleaved rounds:

```
threshold inside the scan   168,977 ms
threshold outside the scan       823 ms      206x
```

It is not the statement size: a 42.6 KB statement without the threshold ran in 827 ms while a
21.4 KB statement carrying one took 70,788 ms.

## Why this is a translation artifact, not a design choice

The comment above the call site records the reasoning:

> Similarity threshold goes straight in the ANN scan's WHERE (relies on the 26.07.4 fix #964 — on
> <26.07.4 a vector-op predicate here silently emptied the result and had to be applied outside the
> scan).

So the predicate was moved *into* the scan to fix a silent-empty bug, and the fix worked. What was
not re-examined is that ES — the backend this connector's semantics were derived from — never had
the threshold constrain the search in the first place.

## Two knobs SereneDB exposes that the connector does not use

```
sdb_ivf_search_nprobe   default 8     IVF cluster lists scanned per query
sdb_rerank_factor       default 4     exact re-score pool = ceil(factor * k)
```

`nprobe` is the direct analogue of ES's `num_candidates` — the recall knob — and RAGFlow already
computes a `num_candidates` value that is then discarded. Measured recall@10 against an exhaustive
scan, on the worst of three probe queries:

```
nprobe=8    4/10      <- default
nprobe=32   8/10
nprobe=128  9/10
nprobe=512  9/10
```

`sdb_rerank_factor=4` means a top-10 query exactly re-scores only 40 candidates, which is the more
likely cause of the 9/10 plateau than quantisation.

## The confound that invalidates most latency numbers

`sdb_metrics` on this instance:

```
num_segments                48
index_size                  221,614,014,434   (206 GiB)
avg_consolidation_time_ms   36,294            <-- 36 s per consolidation
avg_commit_time_ms          121
```

Index consolidation runs continuously while ingesting, reading ~4 GB/s. The same query measured
716 ms and 12,192 ms depending on whether a consolidation was in flight. **Any latency figure taken
during ingest is measuring merge I/O.** Recall figures are unaffected.

## Resolution (2026-09-09)

The threshold was removed from the SQL in all three places it appeared:

| file | change |
|---|---|
| `rag/utils/serenedb_conn.py` | dropped from the vector search and the fusion CTE |
| `internal/engine/serenedb/search.go` | dropped from `buildVectorSQL` and `buildFusionSQL` |
| `internal/engine/serenedb/serenedb_test.go` | `TestBuildVectorSQLThresholdInWhere` → `…NotInWhere` |

The Go path had the identical defect and a test asserting the wrong SQL was generated — the
assumption had propagated into three places.

Removing it is a **correctness** fix, not only a speed one. `rag/nlp/search.py` re-applies the
threshold after retrieval, against the **hybrid** score:

```python
post_threshold = 0.0 if vector_similarity_weight <= 0 else similarity_threshold
valid_idx = [int(i) for i in sorted_idx if sim_np[i] >= post_threshold]
```

The SQL predicate filtered **pure vector similarity**, so it could discard a chunk with a strong
term match that Elasticsearch would have returned.

Measured after the fix, live, at RAGFlow's real `k=1024`, with consolidation still running:

```
                                  before        after (warm)
superconducting qubit decoherence  104,095 ms    325 / 429 ms
transformer attention mechanism     ~104,000 ms   455 / 294 ms
```

Recall@10 against an exhaustive baseline is now bounded by IVF rather than by the scan:
88% at nprobe=8, 93% at 32, 97% at 128.

## Where the CPU actually goes

`perf record -F 199 -g`, 38,909 samples over 60 s on the live instance:

| category | share |
|---|---|
| DuckDB column decompression (`AlpRD`, `FSST`, FastPFOR bitpacking, checksum) | **~37%** |
| kernel (page-fault / IO path) | ~16% |
| BLAS inner product (`sgemm_kernel`, `knn_inner_product`) | ~11% |
| memmove / memset | ~10% |
| iresearch postings **write** (consolidation) | ~2.5% |

Three CPU-seconds are spent unpacking stored bytes for every one spent comparing vectors, and 66%
of active threads sample in `folio_wait_bit_common` — blocked on page reads. The cost is set by how
many rows get touched and decompressed, which is exactly what the radius predicate controlled.

## Storage findings, not yet acted on

* The vector is stored **twice** — `q_1024_vec` and `q_1024_vec_n`, 43.87 billion floats each,
  ~304 GB of the 422 GB `engine_duckdb`. Only the normalized column is searched.
* `ALPRD` on the vector columns achieves roughly **1.16x** (~152 GB stored vs 175.5 GB raw) while
  costing ~10% of all CPU to decompress. SereneDB accepts a per-column `compression 'uncompressed'`
  index option.
* Roughly half the values in each vector column sit in `Constant` segments — unexplained, and worth
  understanding before any rebuild.
* `vec_threshold` / `pm.vecThreshold` are now parsed but unused in both languages: dead surface.
