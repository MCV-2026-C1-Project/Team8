"""Compact experiment artifacts and separate, exact Week 3 comparison."""
import csv
import json
from pathlib import Path
from time import perf_counter
from descriptor_cli import load_descriptor_document
from descriptors import list_images
from filter_noise import query_preprocessor
from retrieval import retrieve_multi_config, source_identity
from local_development import load_development, align_boxes
from local_evaluation import evaluate_local, selection_key
from local_experiment_config import canonical


def compare_week3(config_path, query_dir, db_dir, cache_dir, segmentation, *, query_limit=None, k=5):
    if config_path is None:
        return dict(status="unavailable",error="No exported Week 3 configuration supplied")
    try:
        start=perf_counter(); config,measure=load_descriptor_document(config_path)
        document=json.loads(Path(config_path).read_text(encoding="utf-8"))
        provenance=document.get("provenance",{})
        query_preprocessor(provenance.get("query_filter"))  # Preflight selective dependency.
        paths,ids=list_images(query_dir); labels,frames=load_development(query_dir,paths,ids)
        diagnostics=[]
        qids,rankings=retrieve_multi_config(config,measure,segmentation,db_dir,query_dir,cache_dir,k,
            query_filter=provenance.get("query_filter"),crop_inset=provenance.get("crop_inset",0.),
            fusion=provenance.get("fusion"),diagnostics=diagnostics)
        if qids.tolist() != ids:
            raise ValueError("Week 3 query order changed")
        # Existing Week 3 API processes the full set; subset limits only evaluation.
        rankings,labels,paths,diagnostics=rankings[:query_limit],labels[:query_limit],paths[:query_limit],diagnostics[:query_limit]
        alignments=[align_boxes([p["bbox"] for p in row["paintings"]],frames[path.name]) for row,path in zip(diagnostics,paths)]
        return dict(status="ok", configuration=document, segmentation=segmentation,
                    metrics=evaluate_local(labels,rankings,rankings,alignments),runtime=perf_counter()-start,
                    predictions=rankings,alignment=alignments,
                    scope="development_subset" if len(paths)<len(ids) else "full_development_set",
                    note="Exact Week 3 retrieval; no unknown rejection; full inference even for evaluation subsets")
    except (OSError,ValueError,ModuleNotFoundError) as error:
        return dict(status="unavailable",error=str(error),configuration_path=str(config_path))


def write_outputs(output, run, query_dir, db_dir, baseline=None, diagnostics=False):
    output=Path(output)
    def save(name,value):
        with (output/name).open("x",encoding="utf-8") as stream:
            json.dump(value,stream,indent=2,allow_nan=False); stream.write("\n")
    good=sorted((r for r in run["results"] if r["status"]=="ok"),key=selection_key)
    ordered=good+[r for r in run["results"] if r["status"]!="ok"]
    rows=[]
    for result in ordered:
        c=result["config"]; m=result.get("metrics",{}); u=m.get("unknown_detection") or {}
        row=dict(experiment_id=result["id"],status=result["status"],error=result.get("error",""),
                 feature_method=c["feature_method"],feature_params=canonical(c["feature_params"]),
                 query_filter=canonical(c["query_filter"]),matching_filter=c["matching"]["filter"],
                 ratio_threshold=c["matching"]["ratio_threshold"],estimator=c["geometry"]["estimator"],
                 reprojection_threshold=c["geometry"]["reprojection_threshold"],scoring=c["scoring"]["mode"],
                 coverage_grid=canonical(c["scoring"]["coverage_grid"]),min_inliers=c["rejection"]["min_inliers"],
                 runtime=result["runtime"],missing=m.get("missing"),extra=m.get("extra"),
                 unknown_precision=u.get("precision"),unknown_recall=u.get("recall"),unknown_f1=u.get("f1"),
                 effective_config=canonical(c))
        for metric in ("known_ranking_map","known_after_rejection_map","overall_map","detection_aware_map"):
            for k in ("1","5"):
                row[metric+k]=m.get(metric,{}).get(k)
        rows.append(row)
    with (output/"results.csv").open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    best=run["best"]
    if best:
        save("best_config.json",best["config"])
        save("predictions.json",dict(experiment_id=best["id"],scope=run["scope"],queries=best["predictions"]))
    if diagnostics:
        save("diagnostics.json",[{k:r[k] for k in ("id","alignment","noise_decisions","threshold_selection","timings")}
                                 for r in good])
    if baseline is not None:
        save("week3_baseline.json",baseline)
    warnings=[r["error"] for r in ordered if r["status"]!="ok"]
    if baseline and baseline["status"]!="ok": warnings.append("Week 3 comparison: "+baseline["error"])
    summary=dict(dataset=str(Path(query_dir).resolve()),scope=run["scope"],images=run["images"],paintings=run["paintings"],
                 configurations=len(rows),successful=len(good),best_id=best["id"] if best else None,
                 best_config=best["config"] if best else None,metrics=best["metrics"] if best else None,
                 selection_policy="overall mAP@1, aligned unknown F1, overall mAP@5 descending; stage runtime, stable ID ascending",
                 provenance="QSD1-W4 development-selected; not independent test performance",
                 counts=run["counts"],wall_seconds=run["wall_seconds"],runtime_policy=run["runtime_policy"],
                 opencv_version=run["opencv_version"],warnings=warnings,
                 outputs=[p.name for p in output.iterdir() if p.is_file()])
    save("summary.json",summary)
    save("manifest.json",dict(query_source=source_identity(query_dir),database_source=source_identity(db_dir),
                              experiment_ids=[r["id"] for r in ordered],opencv_version=run["opencv_version"]))
    return summary
