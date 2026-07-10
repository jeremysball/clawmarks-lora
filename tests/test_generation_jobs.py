# tests/test_generation_jobs.py
import random
from clawmarks.search.driver import build_generation_jobs


def test_batch_splits_by_explore_fraction():
    random.seed(0)
    elite = {"prompt_name": "style_x", "prompt": "trentbuckle style, x", "strength": 1.0,
             "cfg": 7.0, "tag": "gen1_explore_0_seed1"}
    jobs = build_generation_jobs(
        gen_idx=2, subjects=["a subject"], textures=["a texture"],
        elites=[elite], user_picks=[], batch_size=20, explore_fraction=0.85,
    )
    n_explore = sum(1 for j in jobs if j["category"] == "r2_explore")
    n_exploit = sum(1 for j in jobs if j["category"] == "r2_exploit")
    assert n_explore == 17
    assert n_exploit == 3
    assert n_explore + n_exploit == 20


def test_fifty_fifty_split_matches_round_one_behavior():
    random.seed(0)
    elite = {"prompt_name": "style_x", "prompt": "trentbuckle style, x", "strength": 1.0,
             "cfg": 7.0, "tag": "gen1_explore_0_seed1"}
    jobs = build_generation_jobs(
        gen_idx=2, subjects=["a subject"], textures=["a texture"],
        elites=[elite], user_picks=[], batch_size=20, explore_fraction=0.5,
    )
    n_explore = sum(1 for j in jobs if j["category"] == "r2_explore")
    n_exploit = sum(1 for j in jobs if j["category"] == "r2_exploit")
    assert n_explore == 10
    assert n_exploit == 10


def test_exploit_jobs_prefer_user_picks_over_elites():
    pick = {"prompt_name": "style_pick", "prompt": "trentbuckle style, pick", "strength": 1.2,
            "cfg": 6.0, "tag": "gen1_explore_1_seed2"}
    elite = {"prompt_name": "style_elite", "prompt": "trentbuckle style, elite", "strength": 1.0,
             "cfg": 7.0, "tag": "gen1_explore_2_seed3"}
    jobs = build_generation_jobs(
        gen_idx=3, subjects=["a subject"], textures=["a texture"],
        elites=[elite], user_picks=[pick], batch_size=4, explore_fraction=0.0,
    )
    assert all(j["prompt_name"] == "style_pick" for j in jobs)


def test_no_elites_and_no_picks_produces_only_explore_jobs():
    jobs = build_generation_jobs(
        gen_idx=1, subjects=["a subject"], textures=["a texture"],
        elites=[], user_picks=[], batch_size=5, explore_fraction=0.85,
    )
    assert all(j["category"] == "r2_explore" for j in jobs)
    assert len(jobs) == 5
