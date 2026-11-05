"""Bounded pair-wise evidence reuse for exhaustive Week 4 development experiments."""
import copy
from collections import defaultdict
from dataclasses import dataclass, replace
from time import perf_counter
import cv2
from descriptors import list_images, load_image
from filter_noise import query_preprocessor
from local_feature_cache import load_or_compute_database_features
from local_feature_queries import prepare_painting_crops
from painting_detection import detect_paintings
from local_matching import tentative_matches, filter_matches
from local_geometry import verify_geometry
from local_retrieval import score_verification, rank_candidates
from local_rejection import painting_evidence, predict, select_threshold
from local_development import load_development, align_boxes
from local_evaluation import evaluate_local, selection_key
from local_experiment_config import canonical


@dataclass(frozen=True)
class VerificationCount:
    success: bool
    inlier_count: int


@dataclass
class RankedEvidence:
    database_id: int
    verification: VerificationCount
    scores: dict


def run_local_experiments(jobs, query_dir, db_dir, cache_dir, *, query_limit=None, progress=print):
    """Jobs come from expand_local_experiments; one DB config and one image live at a time.

    Runtime per job charges each stage it requires once, even if reused; measured
    wall time and actual stage-call counts are also returned. No persisted pair cache.
    """
    start = perf_counter()
    if not jobs:
        raise ValueError("No experiments selected")
    if query_limit is not None and (isinstance(query_limit,bool) or not isinstance(query_limit,int) or query_limit < 1):
        raise ValueError("query_limit must be a positive integer")
    paths, ids = list_images(query_dir)
    labels, frames = load_development(query_dir,paths,ids)
    paths, ids, labels = paths[:query_limit], ids[:query_limit], labels[:query_limit]
    db_paths, db_ids = list_images(db_dir)
    if not db_paths or any(j["config"]["retrieval"]["k"] > len(db_ids) for j in jobs):
        raise ValueError("Complete database must contain at least retrieval.k paintings")
    allowed = set(db_ids)
    if any(label != -1 and label not in allowed for row in labels for label in row):
        raise ValueError("Development label is absent from the museum database")
    counts = dict(database_loads=0, detections=0, query_extractions=0, forward_matches=0, verifications=0, filters=0)
    results, active = [], []
    for job in jobs:
        policy = job["config"]["query_filter"]
        try:
            query_preprocessor(policy)  # Fail missing selective dependencies before vision work.
        except ModuleNotFoundError as error:
            results.append(dict(id=job["id"], config=job["config"], status="error", error=str(error), runtime=0.))
            progress(f"{job['id']}: unavailable query filter: {error}")
            continue
        active.append(job)
    groups = defaultdict(list)
    for job in active:
        c=job["config"]
        groups[canonical([c["feature_method"],c["feature_params"]])].append(job)
    # Retain only lightweight detection boxes across feature groups, never image-sized masks.
    detections = {}
    best = None
    for group in groups.values():
        config = group[0]["config"]
        before=perf_counter()
        database=load_or_compute_database_features(db_dir,config["feature_method"],cache_dir,**config["feature_params"])
        db_time=perf_counter()-before; counts["database_loads"]+=1
        progress(f"{config['feature_method']}: database cache {database.cache_status}; sharing features across {len(group)} configurations")
        timings={j["id"]:dict(extraction=db_time,matching=0.,verification=0.,scoring=0.) for j in group}
        evidence={j["id"]:[] for j in group}; alignments={j["id"]:[] for j in group}
        noise={j["id"]:[] for j in group}
        for path,qid in zip(paths,ids):
            image=load_image(path)
            inputs=defaultdict(list)
            for job in group:
                c=job["config"]
                inputs[canonical([c["segmentation"],c["crop"],c["query_filter"]])].append(job)
            for input_jobs in inputs.values():
                c=input_jobs[0]["config"]
                detection_key=(path.name,canonical(c["segmentation"]))
                if detection_key not in detections:
                    before=perf_counter()
                    _, paintings=detect_paintings(image,c["segmentation"])
                    detections[detection_key]=([replace(p,mask=None) for p in paintings],perf_counter()-before)
                    counts["detections"]+=1
                paintings,detection_time=detections[detection_key]
                decisions=[]
                before=perf_counter()
                preprocess=query_preprocessor(c["query_filter"],diagnostics=decisions)
                crops=prepare_painting_crops(image,qid,c["segmentation"],crop_inset=c["crop"]["inset"],
                                             preprocess=preprocess,detections=paintings)
                prep_time=perf_counter()-before+detection_time
                alignment=align_boxes([p.detection_bbox for p in crops],frames[path.name])
                if preprocess is None:
                    decisions=[dict(filtered=False,reason="disabled") for _ in crops]
                for job in input_jobs:
                    evidence[job["id"]].append([]); alignments[job["id"]].append(alignment)
                    noise[job["id"]].append(dict(query_id=qid,decisions=decisions))
                    timings[job["id"]]["extraction"]+=prep_time
                for crop in crops:
                    before=perf_counter(); query=crop.extract(c["feature_method"],**c["feature_params"]).features
                    extraction=perf_counter()-before; counts["query_extractions"]+=1
                    candidates={j["id"]:[] for j in input_jobs}
                    mutual=any(j["config"]["matching"]["filter"] == "ratio_mutual" for j in input_jobs)
                    for image_id,db_features in zip(database.ids,database.features):
                        before=perf_counter(); tentative=tentative_matches(query,db_features,compute_reverse=mutual)
                        matching_time=perf_counter()-before; counts["forward_matches"]+=1
                        filtered,verified,scored={},{},{}
                        for job in input_jobs:
                            jc=job["config"]; jid=job["id"]
                            mk=canonical(jc["matching"]); gk=(mk,canonical(jc["geometry"]))
                            if mk not in filtered:
                                before=perf_counter(); matches=filter_matches(tentative,jc["matching"])
                                filtered[mk]=(matches,perf_counter()-before); counts["filters"]+=1
                            matches,filter_time=filtered[mk]
                            if gk not in verified:
                                before=perf_counter(); v=verify_geometry(matches,jc["geometry"])
                                verified[gk]=(v,perf_counter()-before); counts["verifications"]+=1
                            v,geometry_time=verified[gk]
                            sk=(gk,canonical(jc["scoring"]["coverage_grid"]))
                            if sk not in scored:
                                before=perf_counter(); score=score_verification(v,jc["scoring"])
                                scored[sk]=(score,perf_counter()-before)
                            score,score_time=scored[sk]
                            candidates[jid].append(RankedEvidence(image_id,VerificationCount(v.success,v.inlier_count),score))
                            timings[jid]["matching"]+=matching_time+filter_time
                            timings[jid]["verification"]+=geometry_time; timings[jid]["scoring"]+=score_time
                    for job in input_jobs:
                        jid=job["id"]; jc=job["config"]
                        ranked=rank_candidates(candidates[jid],jc["scoring"]["mode"],jc["retrieval"]["k"])
                        evidence[jid][-1].append(painting_evidence(ranked,jc["retrieval"]["k"]))
                        timings[jid]["extraction"]+=extraction
            progress(f"{config['feature_method']} photo {qid}: shared matching/geometry across {len(group)} configurations")
        for job in group:
            jid=job["id"]; jc=copy.deepcopy(job["config"])
            aligned_evidence,aligned_labels=[],[]
            for row,truth,alignment in zip(evidence[jid],labels,alignments[jid]):
                for label,m in zip(truth,alignment["matches"]):
                    if m is not None:
                        aligned_evidence.append(row[m]); aligned_labels.append(label)
            selection=None
            if job["evaluation"]["select_rejection_threshold"]:
                if not aligned_labels:
                    results.append(dict(id=jid,config=jc,status="error",error="No geometrically aligned paintings for threshold selection",runtime=sum(timings[jid].values())))
                    continue
                selection=select_threshold(aligned_evidence,aligned_labels)
                jc["rejection"]=dict(mode="min_inliers",min_inliers=selection["min_inliers"])
            rankings=[[list(p.top_ids) for p in row] for row in evidence[jid]]
            predictions=[[predict(p,jc["rejection"]) for p in row] for row in evidence[jid]]
            metrics=evaluate_local(labels,rankings,predictions,alignments[jid],job["evaluation"]["k_values"])
            result=dict(id=jid,config=jc,status="ok",metrics=metrics,runtime=sum(timings[jid].values()),timings=timings[jid],
                        cache_status=database.cache_status,threshold_selection=selection,
                        predictions=[dict(query_id=qid,image=p.name,rankings=raw,predictions=pred) for qid,p,raw,pred in zip(ids,paths,rankings,predictions)],
                        alignment=alignments[jid],noise_decisions=noise[jid])
            results.append(result)
            if best is None or selection_key(result) < selection_key(best): best=result
            progress(f"Completed {len(results)}/{len(jobs)} {jid} {jc['feature_method']}/"
                     f"{jc['matching']['filter']}/{jc['geometry']['estimator']}/{jc['scoring']['mode']}: mAP@1={metrics['overall_map']['1']:.4f}, "
                     f"unknown F1={metrics['unknown_detection']['f1'] if metrics['unknown_detection'] else 'undefined'}, "
                     f"threshold={jc['rejection']['min_inliers']}; best={best['id']}")
    return dict(results=results,best=best,counts=counts,wall_seconds=perf_counter()-start,
                query_ids=ids,images=len(paths),paintings=sum(map(len,labels)),
                scope="development_subset" if query_limit is not None and query_limit < len(list_images(query_dir)[0]) else "full_development_set",
                runtime_policy="per-configuration sum of required measured stages, including shared work; wall time reported separately",
                opencv_version=cv2.__version__)
