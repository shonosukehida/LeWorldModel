from datetime import datetime
from zoneinfo import ZoneInfo

from diamond_run import prepare_run


def test_collision_and_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    now = datetime.fromisoformat("2026-11-18T03:30:00+00:00").astimezone(ZoneInfo("Asia/Tokyo"))
    def config(name=None):
        return dict(output=None, wandb=dict(name=name))
    first, second = config(), config("flip-mug-diamond")
    assert str(prepare_run(first,None,{},now)) == "outputs/diamond/20261118_123000"
    assert first["run_name"] == "diamond_20261118_123000"
    assert first["started_at"] == "2026-11-18T12:30:00+09:00"
    assert prepare_run(second,None,{},now).name == "20261118_123000_02"
    assert second["run_name"] == "flip-mug-diamond_20261118_123000_02"
    assert first["run_id"] != second["run_id"]
    resumed = config("ignored")
    path = prepare_run(resumed,second,{},now.replace(day=19))
    assert path.name == "20261118_123000_resume"
    for key in ("run_id","started_at","run_timestamp","run_name"):
        assert resumed[key] == second[key]


def test_legacy_online_identity(tmp_path):
    import pytest
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    cfg = dict(output=str(tmp_path/"new"), wandb=dict(name="override"))
    previous = dict(wandb=dict(name="original",project="saved"))
    with pytest.warns(UserWarning, match="no run timestamp"):
        prepare_run(cfg, previous, {"run_id":"online123"}, now)
    assert cfg["run_name"] == "original" and cfg["run_id"] == "online123"
    assert cfg["wandb"]["project"] == "saved"
